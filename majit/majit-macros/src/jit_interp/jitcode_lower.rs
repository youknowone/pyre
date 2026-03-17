use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    BinOp, Block, Expr, ExprAssign, ExprBinary, ExprCall, ExprCast, ExprIf, ExprLit,
    ExprMethodCall, ExprParen, ExprPath, ExprReference, ExprUnary, FnArg, Ident, ItemFn, Lit,
    Local, Pat, Path, ReturnType, Stmt, Type, UnOp,
};

// ── LowererConfig ────────────────────────────────────────────────────

/// Configuration for storage-aware JitCode lowering.
///
/// Built from `JitInterpConfig` at proc-macro time and passed to the Lowerer
/// to recognize compound storage methods, I/O shims, and selector assignments.
pub struct LowererConfig {
    /// Normalized pool expression (e.g., "state.storage").
    pool_str: String,
    /// Normalized selector expression (e.g., "state.selected").
    selector_str: String,
    /// Normalized "pool.get_mut(selector)" receiver pattern.
    pool_get_mut_sel: String,
    /// Method name → IR OpCode ident (e.g., "add" → IntAdd).
    binops: HashMap<String, Ident>,
    /// Normalized I/O func path → shim ident (e.g., "aheui_io::write_number" → jit_write_number).
    io_shims: Vec<(String, Ident)>,
    /// Normalized helper func path → explicit or inferred call policy.
    calls: Vec<(String, CallPolicySpec)>,
    /// Whether top-level traced calls should auto-infer helper policy.
    auto_calls: bool,
    /// Virtualizable variable name (normalized, e.g., "frame").
    /// RPython jtransform.py: `is_virtualizable_getset()` uses this to check
    /// if a field access target is the virtualizable variable.
    vable_var: Option<String>,
    /// Field name → (field_index, field_type_str).
    /// RPython: `vinfo.static_field_to_extra_box[fieldname]` → index.
    vable_fields: HashMap<String, (usize, String)>,
    /// Array name → (array_index, item_type_str).
    /// RPython: `vinfo.array_field_counter[fieldname]` → index.
    vable_arrays: HashMap<String, (usize, String)>,
}

const MAX_HELPER_CALL_ARITY: usize = 16;

pub(crate) struct InlineHelperJitCode {
    pub body: TokenStream,
    pub return_reg: u16,
    pub return_kind: InlineReturnKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InlineReturnKind {
    Int,
    Ref,
    Float,
}

#[derive(Clone)]
enum CallPolicySpec {
    Explicit(Ident),
    Infer,
}

#[derive(Clone, Copy)]
enum InferenceFailureMode {
    ReturnNone,
    Panic,
}

impl LowererConfig {
    pub fn new(
        pool_expr: &Expr,
        selector_expr: &Expr,
        binops: &[(Ident, Ident)],
        io_shims: &[(Path, Ident)],
        calls: &[(Path, Option<Ident>)],
        auto_calls: bool,
        vable_decl: Option<&crate::jit_interp::VirtualizableDecl>,
    ) -> Self {
        let pool_str = normalize(&quote!(#pool_expr).to_string());
        let selector_str = normalize(&quote!(#selector_expr).to_string());
        let pool_get_mut_sel = format!("{}.get_mut({})", pool_str, selector_str);
        let binops = binops
            .iter()
            .map(|(m, o)| (m.to_string(), o.clone()))
            .collect();
        let io_shims = io_shims
            .iter()
            .map(|(p, s)| (normalize(&quote!(#p).to_string()), s.clone()))
            .collect();
        let calls = calls
            .iter()
            .map(|(p, k)| {
                let spec = match k {
                    Some(kind) => CallPolicySpec::Explicit(kind.clone()),
                    None => CallPolicySpec::Infer,
                };
                (normalize(&quote!(#p).to_string()), spec)
            })
            .collect();
        let (vable_var, vable_fields, vable_arrays) = if let Some(decl) = vable_decl {
            let var = Some(decl.var_name.to_string());
            let fields = decl
                .fields
                .iter()
                .enumerate()
                .map(|(i, f)| (f.name.to_string(), (i, f.field_type.to_string())))
                .collect();
            let arrays = decl
                .arrays
                .iter()
                .enumerate()
                .map(|(i, a)| (a.name.to_string(), (i, a.item_type.to_string())))
                .collect();
            (var, fields, arrays)
        } else {
            (None, HashMap::new(), HashMap::new())
        };
        Self {
            pool_str,
            selector_str,
            pool_get_mut_sel,
            binops,
            io_shims,
            calls,
            auto_calls,
            vable_var,
            vable_fields,
            vable_arrays,
        }
    }
}

fn normalize(s: &str) -> String {
    s.replace(' ', "")
}

fn normalize_expr(expr: &Expr) -> String {
    normalize(&quote!(#expr).to_string())
}

fn unwrap_ref_expr(expr: &Expr) -> &Expr {
    match expr {
        Expr::Reference(ExprReference { expr, .. }) => expr,
        _ => expr,
    }
}

// ── Lowerer ──────────────────────────────────────────────────────────

#[derive(Clone)]
enum BindingKind {
    Int,
    Ref,
    Float,
}

#[derive(Clone)]
struct Binding {
    reg: u16,
    kind: BindingKind,
    depends_on_stack: bool,
}

struct Lowerer<'c> {
    bindings: HashMap<String, Binding>,
    statements: Vec<TokenStream>,
    next_reg: u16,
    next_label: u16,
    config: Option<&'c LowererConfig>,
    call_policies: Vec<(String, CallPolicySpec)>,
    inference_failure_mode: InferenceFailureMode,
    auto_calls: bool,
}

impl<'c> Lowerer<'c> {
    fn new(config: Option<&'c LowererConfig>) -> Self {
        let call_policies = config.map(|cfg| cfg.calls.clone()).unwrap_or_default();
        Self::new_with_call_policies(config, call_policies, InferenceFailureMode::ReturnNone)
    }

    fn new_with_call_policies(
        config: Option<&'c LowererConfig>,
        call_policies: Vec<(String, CallPolicySpec)>,
        inference_failure_mode: InferenceFailureMode,
    ) -> Self {
        Self {
            bindings: HashMap::new(),
            statements: Vec::new(),
            next_reg: 0,
            next_label: 0,
            config,
            call_policies,
            inference_failure_mode,
            auto_calls: config.map(|cfg| cfg.auto_calls).unwrap_or(false),
        }
    }

    fn alloc_reg(&mut self) -> u16 {
        let reg = self.next_reg;
        self.next_reg = self.next_reg.saturating_add(1);
        reg
    }

    fn alloc_label(&mut self) -> syn::Ident {
        let label = self.next_label;
        self.next_label = self.next_label.saturating_add(1);
        format_ident!("__jit_label_{label}")
    }

    fn inference_failure_tokens(&self, message: &str) -> TokenStream {
        match self.inference_failure_mode {
            InferenceFailureMode::ReturnNone => quote! { return None; },
            InferenceFailureMode::Panic => {
                let message = message.to_string();
                quote! { panic!(#message); }
            }
        }
    }

    fn resolve_call_policy(&self, func: &Expr) -> Option<CallPolicySpec> {
        let func_str = normalize_expr(func);
        if let Some((_, policy)) = self
            .call_policies
            .iter()
            .find(|(path, _)| *path == func_str)
        {
            return Some(policy.clone());
        }
        match self.inference_failure_mode {
            InferenceFailureMode::Panic => helper_policy_path(func).map(|_| CallPolicySpec::Infer),
            InferenceFailureMode::ReturnNone => {
                if self.auto_calls {
                    helper_policy_path(func).map(|_| CallPolicySpec::Infer)
                } else {
                    None
                }
            }
        }
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> Option<()> {
        match stmt {
            Stmt::Local(local) => {
                if let Some(()) = self.lower_local(local) {
                    return Some(());
                }
                if self.config.is_some() && !self.stmt_modifies_jit_state(stmt) {
                    return Some(());
                }
                None
            }
            Stmt::Expr(expr, _) => {
                if let Some(()) = self.lower_expr_stmt(expr) {
                    return Some(());
                }
                if self.config.is_some() && !self.stmt_modifies_jit_state(stmt) {
                    return Some(());
                }
                None
            }
            Stmt::Item(_) | Stmt::Macro(_) => None,
        }
    }

    fn lower_local(&mut self, local: &Local) -> Option<()> {
        let Pat::Ident(pat_ident) = &local.pat else {
            return None;
        };
        let init = local.init.as_ref()?;

        // Try normal lowering
        if let Some(binding) = self.lower_value_expr(&init.expr) {
            self.bindings.insert(pat_ident.ident.to_string(), binding);
            return Some(());
        }

        // Config-aware: runtime constant (expression not touching storage)
        if self.config.is_some() && !self.expr_touches_storage(&init.expr) {
            let reg = self.alloc_reg();
            let ident = &pat_ident.ident;
            let init_expr = &init.expr;
            self.statements.push(quote! {
                let #ident = #init_expr;
                __builder.load_const_i_value(#reg, #ident as i64);
            });
            self.bindings.insert(
                ident.to_string(),
                Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: false,
                },
            );
            return Some(());
        }

        None
    }

    /// RPython jtransform.py:923 `_rewrite_op_setfield` for virtualizable.
    ///
    /// Recognizes `frame.field_name = value` and emits vable_setfield JitCode.
    fn lower_vable_field_write(&mut self, expr: &Expr) -> Option<()> {
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
        let receiver_str = normalize(&quote!(#field.base).to_string());
        if receiver_str != *vable_var {
            return None;
        }
        let member_name = match &field.member {
            syn::Member::Named(ident) => ident.to_string(),
            _ => return None,
        };
        let &(field_index, _) = config.vable_fields.get(&member_name)?;
        let fi = field_index as u16;
        let binding = self.lower_value_expr(&assign.right)?;
        let src = binding.reg;
        self.statements.push(quote! {
            __builder.vable_setfield(#fi, #src);
        });
        Some(())
    }

    /// RPython jtransform.py:794 `setarrayitem_vable_*`.
    ///
    /// Recognizes `frame.locals_w[i] = val` and emits vable_setarrayitem.
    fn lower_vable_array_write(&mut self, expr: &Expr) -> Option<()> {
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
        let receiver_str = normalize(&quote!(#field.base).to_string());
        if receiver_str != *vable_var {
            return None;
        }
        let member_name = match &field.member {
            syn::Member::Named(ident) => ident.to_string(),
            _ => return None,
        };
        let &(array_index, _) = config.vable_arrays.get(&member_name)?;
        let ai = array_index as u16;

        // Lower index and value
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        let idx_reg = idx_binding.reg;
        let val_binding = self.lower_value_expr(&assign.right)?;
        let val_reg = val_binding.reg;

        self.statements.push(quote! {
            __builder.vable_setarrayitem(#ai, #idx_reg, #val_reg);
        });
        Some(())
    }

    /// RPython jtransform.py:650 `hint_force_virtualizable`.
    ///
    /// Recognizes `hint_force_virtualizable!(frame)` macro invocation.
    fn lower_vable_force(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let _vable_var = config.vable_var.as_ref()?;

        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let seg = mac.mac.path.segments.last()?;
        if seg.ident != "hint_force_virtualizable" {
            return None;
        }
        self.statements.push(quote! {
            __builder.vable_force();
        });
        Some(())
    }

    /// RPython jtransform.py:655 — suppress identity hint function calls.
    ///
    /// `hint_access_directly(frame)` and `hint_fresh_virtualizable(frame)`
    /// are identity functions that return their argument unchanged.
    /// The Lowerer recognizes these calls and lowers the argument directly,
    /// effectively eliminating the hint call.
    fn lower_vable_hint_identity_call(&mut self, expr: &Expr) -> Option<Binding> {
        let call = match expr {
            Expr::Call(c) => c,
            _ => return None,
        };
        let func_name = match &*call.func {
            Expr::Path(p) => {
                let seg = p.path.segments.last()?;
                seg.ident.to_string()
            }
            _ => return None,
        };
        match func_name.as_str() {
            "hint_access_directly" | "hint_fresh_virtualizable" => {
                // These are identity: return lower(arg)
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
    fn lower_vable_hint_suppress(&self, expr: &Expr) -> Option<()> {
        let _config = self.config?;
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let seg = mac.mac.path.segments.last()?;
        match seg.ident.to_string().as_str() {
            "hint_access_directly" | "hint_fresh_virtualizable" => Some(()),
            _ => None,
        }
    }

    fn lower_expr_stmt(&mut self, expr: &Expr) -> Option<()> {
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

        // Config-aware: push_to (non-current storage) BEFORE regular push
        if self.config.is_some() {
            if let Some(()) = self.lower_push_to_stmt(expr) {
                return Some(());
            }
        }

        if let Some(push_arg) = push_call_arg(expr) {
            let binding = self.lower_value_expr(push_arg)?;
            let reg = binding.reg;
            match binding.kind {
                BindingKind::Int => self.statements.push(quote! {
                    __builder.push_i(#reg);
                }),
                BindingKind::Ref => self.statements.push(quote! {
                    __builder.push_r(#reg);
                }),
                BindingKind::Float => self.statements.push(quote! {
                    __builder.push_f(#reg);
                }),
            }
            return Some(());
        }

        if is_pop_discard(expr) {
            self.statements.push(quote! {
                __builder.pop_discard();
            });
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
            if let Some(()) = self.lower_storage_method_stmt(expr) {
                return Some(());
            }
            if let Some(()) = self.lower_io_call_stmt(expr) {
                return Some(());
            }
            if let Some(()) = self.lower_selector_assign(expr) {
                return Some(());
            }
        }

        None
    }

    // ── Config-aware lowering methods ────────────────────────────────

    /// Lower compound storage method: pool.get_mut(sel).add() → inline_call(sub-JitCode)
    fn lower_storage_method_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::MethodCall(mc) = expr else {
            return None;
        };
        let config = self.config?;
        let receiver_str = normalize_expr(&mc.receiver);
        if receiver_str != config.pool_get_mut_sel {
            return None;
        }
        let method_name = mc.method.to_string();

        // Compound binop: pop two, operate, push result
        if let Some(opcode) = config.binops.get(&method_name) {
            let binop_tokens = if opcode.to_string().ends_with("Ovf") {
                quote! {
                    __sub.peek_i(0);
                    __sub.swap_stack();
                    __sub.peek_i(1);
                    __sub.swap_stack();
                    __sub.record_binop_i(2, majit_ir::OpCode::#opcode, 1, 0);
                    __sub.pop_discard();
                    __sub.pop_discard();
                    __sub.push_i(2);
                }
            } else {
                quote! {
                    __sub.pop_i(0);
                    __sub.pop_i(1);
                    __sub.record_binop_i(2, majit_ir::OpCode::#opcode, 1, 0);
                    __sub.push_i(2);
                }
            };
            self.statements.push(quote! {
                {
                    let mut __sub = majit_meta::JitCodeBuilder::new();
                    #binop_tokens
                    let __sub_idx = __builder.add_sub_jitcode(__sub.finish());
                    __builder.inline_call(__sub_idx);
                }
            });
            return Some(());
        }

        if method_name == "dup" && mc.args.is_empty() {
            self.statements.push(quote! { __builder.dup_stack(); });
            return Some(());
        }

        if method_name == "swap" && mc.args.is_empty() {
            self.statements.push(quote! { __builder.swap_stack(); });
            return Some(());
        }

        None
    }

    fn lower_config_call_stmt(&mut self, expr: &Expr) -> Option<()> {
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
        match policy {
            CallPolicySpec::Explicit(kind) => match kind.to_string().as_str() {
                "residual_void" => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.residual_call_void_args(__fn_idx, &[#(#arg_regs),*]);
                        });
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.residual_call_void_typed_args(__fn_idx, #typed_args);
                        });
                    }
                }
                "may_force_void" => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_may_force_void_args(__fn_idx, &[#(#arg_regs),*]);
                        });
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_may_force_void_typed_args(__fn_idx, #typed_args);
                        });
                    }
                }
                "release_gil_void" => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_release_gil_void_args(__fn_idx, &[#(#arg_regs),*]);
                        });
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_release_gil_void_typed_args(__fn_idx, #typed_args);
                        });
                    }
                }
                "loopinvariant_void" => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_loopinvariant_void_args(__fn_idx, &[#(#arg_regs),*]);
                        });
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_loopinvariant_void_typed_args(__fn_idx, #typed_args);
                        });
                    }
                }
                "residual_void_wrapped" => {
                    let policy_path = helper_policy_path(&call.func)?;
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let call_stmt =
                        quote! { __builder.residual_call_void_typed_args(__fn_idx, #typed_args); };
                    self.statements.push(quote! {
                        let (__policy, _inline_builder, __trace_target, __concrete_target) = #policy_path();
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
                        let __fn_idx = __builder.add_call_target(__trace_target, __concrete_target);
                        #call_stmt
                    });
                }
                "may_force_void_wrapped"
                | "release_gil_void_wrapped"
                | "loopinvariant_void_wrapped" => {
                    let policy_path = helper_policy_path(&call.func)?;
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let call_stmt = match kind.to_string().as_str() {
                        "may_force_void_wrapped" => {
                            quote! { __builder.call_may_force_void_typed_args(__fn_idx, #typed_args); }
                        }
                        "release_gil_void_wrapped" => {
                            quote! { __builder.call_release_gil_void_typed_args(__fn_idx, #typed_args); }
                        }
                        "loopinvariant_void_wrapped" => {
                            quote! { __builder.call_loopinvariant_void_typed_args(__fn_idx, #typed_args); }
                        }
                        _ => unreachable!(),
                    };
                    self.statements.push(quote! {
                        let (__policy, _inline_builder, __trace_target, __concrete_target) = #policy_path();
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
                        let __fn_idx = __builder.add_call_target(__trace_target, __concrete_target);
                        #call_stmt
                    });
                }
                _ => return None,
            },
            CallPolicySpec::Infer => {
                let policy_path = helper_policy_path(&call.func)?;
                let typed_args = typed_call_arg_tokens(&arg_bindings);
                let unsupported = self.inference_failure_tokens(
                    "inferred helper policy does not support void calls here",
                );
                self.statements.push(quote! {
                    let (__policy, _inline_builder, __trace_target, __concrete_target) = #policy_path();
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
                    let __fn_idx = __builder.add_call_target(__trace_target, __concrete_target);
                    match __policy {
                        1u8 => {
                            __builder.residual_call_void_typed_args(__fn_idx, #typed_args);
                        }
                        9u8 => {
                            __builder.call_may_force_void_typed_args(__fn_idx, #typed_args);
                        }
                        13u8 => {
                            __builder.call_release_gil_void_typed_args(__fn_idx, #typed_args);
                        }
                        17u8 => {
                            __builder.call_loopinvariant_void_typed_args(__fn_idx, #typed_args);
                        }
                        _ => {
                            #unsupported
                        }
                    }
                });
            }
        }
        Some(())
    }

    /// Lower I/O call: aheui_io::write_number(r, writer) → residual_call_void(shim, r)
    fn lower_io_call_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Call(call) = expr else {
            return None;
        };
        let config = self.config?;
        let func_str = normalize_expr(&call.func);

        for (io_path, shim) in &config.io_shims {
            if func_str == *io_path {
                let arg = unwrap_ref_expr(call.args.first()?);
                let binding = self.lower_value_expr(arg)?;
                let reg = binding.reg;
                self.statements.push(quote! {
                    let __fn_idx = __builder.add_fn_ptr(#shim as *const ());
                    __builder.residual_call_void_args(__fn_idx, &[#reg]);
                });
                return Some(());
            }
        }

        None
    }

    /// Lower selector changes into jitcode when the RHS is trace-time constant.
    fn lower_selector_assign(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Assign(ExprAssign { left, right, .. }) = expr else {
            return None;
        };
        let config = self.config?;
        let lhs_str = normalize_expr(left);
        if lhs_str != config.selector_str {
            return None;
        }
        if self.expr_touches_storage(right) || self.expr_uses_stack_bindings(right) {
            return None;
        }
        self.statements.push(quote! {
            let __selected_value = (#right) as i64;
            let __selected_const_idx = __builder.add_const_i(__selected_value);
            __builder.set_selected(__selected_const_idx);
        });
        Some(())
    }

    /// Lower push to non-current storage: pool.get_mut(target).push(r) → push_to(r, target)
    fn lower_push_to_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::MethodCall(mc) = expr else {
            return None;
        };
        if mc.method != "push" || mc.args.len() != 1 {
            return None;
        }
        let config = self.config?;
        let target_arg = extract_pool_get_mut_arg(&mc.receiver, config)?;
        let target_str = normalize_expr(&target_arg);
        // If target == selector, this is a regular push (handled elsewhere)
        if target_str == config.selector_str {
            return None;
        }
        let arg = &mc.args[0];
        let binding = self.lower_value_expr(arg)?;
        if !matches!(binding.kind, BindingKind::Int) {
            return None;
        }
        let reg = binding.reg;
        self.statements.push(quote! {
            __builder.push_to(#reg, (#target_arg) as u16);
        });
        Some(())
    }

    /// Check if a statement modifies JIT-visible state (storage writes).
    fn stmt_modifies_jit_state(&self, stmt: &Stmt) -> bool {
        let config = match self.config {
            Some(c) => c,
            None => return true,
        };
        let s = normalize(&quote!(#stmt).to_string());
        // Storage mutation: pool.get_mut(...)
        s.contains(&format!("{}.get_mut", config.pool_str))
    }

    /// Check if an expression touches the storage pool.
    fn expr_touches_storage(&self, expr: &Expr) -> bool {
        let config = match self.config {
            Some(c) => c,
            None => return true,
        };
        let s = normalize(&quote!(#expr).to_string());
        s.contains(&config.pool_str)
    }

    fn expr_uses_stack_bindings(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Path(ExprPath { path, .. }) => path
                .get_ident()
                .and_then(|ident| self.bindings.get(&ident.to_string()))
                .is_some_and(|binding| binding.depends_on_stack),
            Expr::Assign(ExprAssign { left, right, .. }) => {
                self.expr_uses_stack_bindings(left) || self.expr_uses_stack_bindings(right)
            }
            Expr::Binary(ExprBinary { left, right, .. }) => {
                self.expr_uses_stack_bindings(left) || self.expr_uses_stack_bindings(right)
            }
            Expr::Call(ExprCall { func, args, .. }) => {
                self.expr_uses_stack_bindings(func)
                    || args.iter().any(|arg| self.expr_uses_stack_bindings(arg))
            }
            Expr::Cast(ExprCast { expr, .. })
            | Expr::Paren(ExprParen { expr, .. })
            | Expr::Unary(ExprUnary { expr, .. }) => self.expr_uses_stack_bindings(expr),
            Expr::If(ExprIf {
                cond,
                then_branch,
                else_branch,
                ..
            }) => {
                self.expr_uses_stack_bindings(cond)
                    || then_branch
                        .stmts
                        .iter()
                        .any(|stmt| self.stmt_uses_stack_bindings(stmt))
                    || else_branch
                        .as_ref()
                        .is_some_and(|(_, expr)| self.expr_uses_stack_bindings(expr))
            }
            _ => false,
        }
    }

    fn stmt_uses_stack_bindings(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr(expr, _) => self.expr_uses_stack_bindings(expr),
            Stmt::Local(local) => local
                .init
                .as_ref()
                .is_some_and(|init| self.expr_uses_stack_bindings(&init.expr)),
            _ => false,
        }
    }

    // ── Core lowering (unchanged logic) ──────────────────────────────

    fn lower_if_stmt(&mut self, expr_if: &ExprIf) -> Option<()> {
        let cond = self.lower_value_expr(&expr_if.cond)?;
        let else_label = self.alloc_label();
        let end_label = self.alloc_label();
        let cond_reg = cond.reg;
        let then_stmts = self.lower_branch_expr(&Expr::Block(syn::ExprBlock {
            attrs: Vec::new(),
            label: None,
            block: expr_if.then_branch.clone(),
        }))?;
        let else_stmts = match expr_if.else_branch.as_ref() {
            Some((_, else_expr)) => self.lower_branch_expr(else_expr)?,
            None => Vec::new(),
        };

        self.statements.push(quote! {
            let #else_label = __builder.new_label();
        });
        self.statements.push(quote! {
            let #end_label = __builder.new_label();
        });
        self.statements.push(quote! {
            __builder.branch_reg_zero(#cond_reg, #else_label);
        });
        self.statements.extend(then_stmts);
        self.statements.push(quote! {
            __builder.jump(#end_label);
        });
        self.statements.push(quote! {
            __builder.mark_label(#else_label);
        });
        self.statements.extend(else_stmts);
        self.statements.push(quote! {
            __builder.mark_label(#end_label);
        });
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
    fn lower_match_stmt(&mut self, expr_match: &syn::ExprMatch) -> Option<()> {
        let discriminant = self.lower_value_expr(&expr_match.expr)?;
        if !matches!(discriminant.kind, BindingKind::Int) {
            return None;
        }

        let end_label = self.alloc_label();
        self.statements.push(quote! {
            let #end_label = __builder.new_label();
        });

        // Separate literal/path arms from the wildcard/default arm.
        let mut guarded_arms = Vec::new();
        let mut default_arm = None;

        for arm in &expr_match.arms {
            match &arm.pat {
                Pat::Wild(_) => {
                    default_arm = Some(&arm.body);
                }
                Pat::Ident(pat_ident) if pat_ident.subpat.is_none() => {
                    // Catch-all binding like `x => ...` treated as default
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
            self.statements.push(quote! {
                let #next_label = __builder.new_label();
            });

            if literals.len() == 1 {
                // Single literal: eq check + branch
                let value = literals[0];
                let const_reg = self.alloc_reg();
                let eq_reg = self.alloc_reg();
                self.statements.push(quote! {
                    __builder.load_const_i_value(#const_reg, #value);
                });
                self.statements.push(quote! {
                    __builder.record_binop_i(#eq_reg, majit_ir::OpCode::IntEq, #disc_reg, #const_reg);
                });
                self.statements.push(quote! {
                    __builder.branch_reg_zero(#eq_reg, #next_label);
                });
            } else {
                // Multiple literals (Or pattern): chain with logical OR
                // (val == lit1) | (val == lit2) | ...
                let first_val = literals[0];
                let first_const_reg = self.alloc_reg();
                let mut or_reg = self.alloc_reg();
                self.statements.push(quote! {
                    __builder.load_const_i_value(#first_const_reg, #first_val);
                });
                self.statements.push(quote! {
                    __builder.record_binop_i(#or_reg, majit_ir::OpCode::IntEq, #disc_reg, #first_const_reg);
                });
                for &lit_val in &literals[1..] {
                    let const_reg = self.alloc_reg();
                    let eq_reg = self.alloc_reg();
                    let new_or_reg = self.alloc_reg();
                    self.statements.push(quote! {
                        __builder.load_const_i_value(#const_reg, #lit_val);
                    });
                    self.statements.push(quote! {
                        __builder.record_binop_i(#eq_reg, majit_ir::OpCode::IntEq, #disc_reg, #const_reg);
                    });
                    self.statements.push(quote! {
                        __builder.record_binop_i(#new_or_reg, majit_ir::OpCode::IntOr, #or_reg, #eq_reg);
                    });
                    or_reg = new_or_reg;
                }
                self.statements.push(quote! {
                    __builder.branch_reg_zero(#or_reg, #next_label);
                });
            }

            let body_stmts = self.lower_branch_expr(body)?;
            self.statements.extend(body_stmts);
            self.statements.push(quote! {
                __builder.jump(#end_label);
            });
            self.statements.push(quote! {
                __builder.mark_label(#next_label);
            });
        }

        // Default arm
        if let Some(default_body) = default_arm {
            let default_stmts = self.lower_branch_expr(default_body)?;
            self.statements.extend(default_stmts);
        }

        self.statements.push(quote! {
            __builder.mark_label(#end_label);
        });
        Some(())
    }

    // ── Loop lowering ────────────────────────────────────────────────

    /// Lower `while cond { body }` to a JitCode branch sequence:
    /// ```text
    /// loop_start:
    ///   eval cond
    ///   branch_reg_zero(cond, loop_end)
    ///   eval body
    ///   jump(loop_start)
    /// loop_end:
    /// ```
    fn lower_while_loop(&mut self, expr_while: &syn::ExprWhile) -> Option<()> {
        let loop_start = self.alloc_label();
        let loop_end = self.alloc_label();

        self.statements.push(quote! {
            let #loop_start = __builder.new_label();
        });
        self.statements.push(quote! {
            let #loop_end = __builder.new_label();
        });
        self.statements.push(quote! {
            __builder.mark_label(#loop_start);
        });

        // Evaluate the condition
        let cond = self.lower_value_expr(&expr_while.cond)?;
        let cond_reg = cond.reg;
        self.statements.push(quote! {
            __builder.branch_reg_zero(#cond_reg, #loop_end);
        });

        // Lower the body, with break targets pointing to loop_end
        let body_stmts = self.lower_loop_body(&expr_while.body, &loop_end, &loop_start)?;
        self.statements.extend(body_stmts);

        // Back-edge jump
        self.statements.push(quote! {
            __builder.jump(#loop_start);
        });
        self.statements.push(quote! {
            __builder.mark_label(#loop_end);
        });
        Some(())
    }

    /// Lower `loop { body }` to a JitCode branch sequence:
    /// ```text
    /// loop_start:
    ///   eval body (break → jump loop_end, continue → jump loop_start)
    ///   jump(loop_start)
    /// loop_end:
    /// ```
    fn lower_loop_expr(&mut self, expr_loop: &syn::ExprLoop) -> Option<()> {
        let loop_start = self.alloc_label();
        let loop_end = self.alloc_label();

        self.statements.push(quote! {
            let #loop_start = __builder.new_label();
        });
        self.statements.push(quote! {
            let #loop_end = __builder.new_label();
        });
        self.statements.push(quote! {
            __builder.mark_label(#loop_start);
        });

        let body_stmts = self.lower_loop_body(&expr_loop.body, &loop_end, &loop_start)?;
        self.statements.extend(body_stmts);

        self.statements.push(quote! {
            __builder.jump(#loop_start);
        });
        self.statements.push(quote! {
            __builder.mark_label(#loop_end);
        });
        Some(())
    }

    /// Lower `for _ in _ { body }`.
    ///
    /// For-loops involve Rust's iterator protocol which cannot be
    /// statically decomposed at proc-macro time. Return `None` so the
    /// arm falls back to opaque (not traced through by the JIT).
    fn lower_for_loop(&mut self, _expr_for: &syn::ExprForLoop) -> Option<()> {
        None
    }

    /// Lower a loop body block, translating `break` → jump to `break_label`
    /// and `continue` → jump to `continue_label`.
    fn lower_loop_body(
        &mut self,
        block: &syn::Block,
        break_label: &syn::Ident,
        continue_label: &syn::Ident,
    ) -> Option<Vec<TokenStream>> {
        let mut nested = Lowerer {
            bindings: self.bindings.clone(),
            statements: Vec::new(),
            next_reg: self.next_reg,
            next_label: self.next_label,
            config: self.config,
            call_policies: self.call_policies.clone(),
            inference_failure_mode: self.inference_failure_mode,
            auto_calls: self.auto_calls,
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
        Some(nested.statements)
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
                self.statements.push(quote! {
                    __builder.jump(#break_label);
                });
                Some(())
            }
            Stmt::Expr(Expr::Continue(_), _) => {
                self.statements.push(quote! {
                    __builder.jump(#continue_label);
                });
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

        let cond = self.lower_value_expr(&expr_if.cond)?;
        let else_label = self.alloc_label();
        let end_label = self.alloc_label();
        let cond_reg = cond.reg;

        self.statements.push(quote! {
            let #else_label = __builder.new_label();
        });
        self.statements.push(quote! {
            let #end_label = __builder.new_label();
        });
        self.statements.push(quote! {
            __builder.branch_reg_zero(#cond_reg, #else_label);
        });

        // Lower then-branch with loop control
        let then_stmts = self.lower_loop_body(&expr_if.then_branch, break_label, continue_label)?;
        self.statements.extend(then_stmts);
        self.statements.push(quote! {
            __builder.jump(#end_label);
        });
        self.statements.push(quote! {
            __builder.mark_label(#else_label);
        });

        // Lower else-branch with loop control
        if let Some((_, else_expr)) = &expr_if.else_branch {
            let else_block = match &**else_expr {
                Expr::Block(block) => &block.block,
                _ => return None,
            };
            let else_stmts = self.lower_loop_body(else_block, break_label, continue_label)?;
            self.statements.extend(else_stmts);
        }

        self.statements.push(quote! {
            __builder.mark_label(#end_label);
        });
        Some(())
    }

    /// Lower a match expression in value position to chained if-else guards
    /// that produce a value.
    fn lower_match_value(&mut self, expr_match: &syn::ExprMatch) -> Option<Binding> {
        let discriminant = self.lower_value_expr(&expr_match.expr)?;
        if !matches!(discriminant.kind, BindingKind::Int) {
            return None;
        }

        let end_label = self.alloc_label();
        let result_reg = self.alloc_reg();
        self.statements.push(quote! {
            let #end_label = __builder.new_label();
        });

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
            self.statements.push(quote! {
                let #next_label = __builder.new_label();
            });

            if literals.len() == 1 {
                let value = literals[0];
                let const_reg = self.alloc_reg();
                let eq_reg = self.alloc_reg();
                self.statements.push(quote! {
                    __builder.load_const_i_value(#const_reg, #value);
                });
                self.statements.push(quote! {
                    __builder.record_binop_i(#eq_reg, majit_ir::OpCode::IntEq, #disc_reg, #const_reg);
                });
                self.statements.push(quote! {
                    __builder.branch_reg_zero(#eq_reg, #next_label);
                });
            } else {
                let first_val = literals[0];
                let first_const_reg = self.alloc_reg();
                let mut or_reg = self.alloc_reg();
                self.statements.push(quote! {
                    __builder.load_const_i_value(#first_const_reg, #first_val);
                });
                self.statements.push(quote! {
                    __builder.record_binop_i(#or_reg, majit_ir::OpCode::IntEq, #disc_reg, #first_const_reg);
                });
                for &lit_val in &literals[1..] {
                    let const_reg = self.alloc_reg();
                    let eq_reg = self.alloc_reg();
                    let new_or_reg = self.alloc_reg();
                    self.statements.push(quote! {
                        __builder.load_const_i_value(#const_reg, #lit_val);
                    });
                    self.statements.push(quote! {
                        __builder.record_binop_i(#eq_reg, majit_ir::OpCode::IntEq, #disc_reg, #const_reg);
                    });
                    self.statements.push(quote! {
                        __builder.record_binop_i(#new_or_reg, majit_ir::OpCode::IntOr, #or_reg, #eq_reg);
                    });
                    or_reg = new_or_reg;
                }
                self.statements.push(quote! {
                    __builder.branch_reg_zero(#or_reg, #next_label);
                });
            }

            let (body_stmts, binding) = self.lower_branch_value_expr(body)?;
            if !matches!(binding.kind, BindingKind::Int) {
                return None;
            }
            depends_on_stack |= binding.depends_on_stack;
            let arm_reg = binding.reg;
            self.statements.extend(body_stmts);
            self.statements.push(quote! {
                __builder.move_i(#result_reg, #arm_reg);
            });
            self.statements.push(quote! {
                __builder.jump(#end_label);
            });
            self.statements.push(quote! {
                __builder.mark_label(#next_label);
            });
        }

        // Default arm
        if let Some(default_body) = default_arm {
            let (default_stmts, default_binding) = self.lower_branch_value_expr(default_body)?;
            if !matches!(default_binding.kind, BindingKind::Int) {
                return None;
            }
            depends_on_stack |= default_binding.depends_on_stack;
            let default_reg = default_binding.reg;
            self.statements.extend(default_stmts);
            self.statements.push(quote! {
                __builder.move_i(#result_reg, #default_reg);
            });
        }

        self.statements.push(quote! {
            __builder.mark_label(#end_label);
        });

        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack,
        })
    }

    /// RPython jtransform.py:832 `rewrite_op_getfield` for virtualizable.
    ///
    /// Recognizes `frame.field_name` where `frame` is the virtualizable variable
    /// and `field_name` is a declared virtualizable field. Emits a vable_getfield
    /// JitCode instruction that will read from virtualizable_boxes at trace time.
    fn lower_vable_field_read(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;

        if let Expr::Field(field) = expr {
            let receiver_str = normalize(&quote!(#field.base).to_string());
            if receiver_str != *vable_var {
                return None;
            }
            let member_name = match &field.member {
                syn::Member::Named(ident) => ident.to_string(),
                _ => return None,
            };

            if let Some(&(field_index, ref field_type)) = config.vable_fields.get(&member_name) {
                let reg = self.alloc_reg();
                let fi = field_index as u16;
                let kind = match field_type.as_str() {
                    "ref" => {
                        self.statements.push(quote! {
                            __builder.vable_getfield_ref(#reg, #fi);
                        });
                        BindingKind::Ref
                    }
                    "float" => {
                        self.statements.push(quote! {
                            __builder.vable_getfield_float(#reg, #fi);
                        });
                        BindingKind::Float
                    }
                    _ => {
                        self.statements.push(quote! {
                            __builder.vable_getfield_int(#reg, #fi);
                        });
                        BindingKind::Int
                    }
                };
                return Some(Binding {
                    reg,
                    kind,
                    depends_on_stack: false,
                });
            }
        }
        None
    }

    /// RPython jtransform.py:760 `getarrayitem_vable_*`.
    ///
    /// Recognizes `frame.locals_w[i]` where `frame` is the virtualizable
    /// variable and `locals_w` is a declared virtualizable array field.
    fn lower_vable_array_read(&mut self, expr: &Expr) -> Option<Binding> {
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
        let receiver_str = normalize(&quote!(#field.base).to_string());
        if receiver_str != *vable_var {
            return None;
        }
        let member_name = match &field.member {
            syn::Member::Named(ident) => ident.to_string(),
            _ => return None,
        };
        let &(array_index, ref item_type) = config.vable_arrays.get(&member_name)?;

        // Lower the index expression to a register
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        let idx_reg = idx_binding.reg;

        let reg = self.alloc_reg();
        let ai = array_index as u16;
        let kind = match item_type.as_str() {
            "ref" => {
                self.statements.push(quote! {
                    __builder.vable_getarrayitem_ref(#reg, #ai, #idx_reg);
                });
                BindingKind::Ref
            }
            "float" => {
                self.statements.push(quote! {
                    __builder.vable_getarrayitem_float(#reg, #ai, #idx_reg);
                });
                BindingKind::Float
            }
            _ => {
                self.statements.push(quote! {
                    __builder.vable_getarrayitem_int(#reg, #ai, #idx_reg);
                });
                BindingKind::Int
            }
        };
        Some(Binding {
            reg,
            kind,
            depends_on_stack: false,
        })
    }

    fn lower_value_expr(&mut self, expr: &Expr) -> Option<Binding> {
        // RPython jtransform.py:832 — virtualizable field read rewrite.
        if let Some(binding) = self.lower_vable_field_read(expr) {
            return Some(binding);
        }
        // RPython jtransform.py:760 — virtualizable array read rewrite.
        if let Some(binding) = self.lower_vable_array_read(expr) {
            return Some(binding);
        }
        // RPython jtransform.py:655 — suppress hint_access_directly(frame) /
        // hint_fresh_virtualizable(frame) function calls as identity.
        // These return the frame unchanged, so lower the argument instead.
        if let Some(binding) = self.lower_vable_hint_identity_call(expr) {
            return Some(binding);
        }

        if is_pop_value(expr) {
            let reg = self.alloc_reg();
            self.statements.push(quote! {
                __builder.pop_i(#reg);
            });
            return Some(Binding {
                reg,
                kind: BindingKind::Int,
                depends_on_stack: true,
            });
        }

        if is_peek_value(expr) {
            let reg = self.alloc_reg();
            self.statements.push(quote! {
                __builder.peek_i(#reg);
            });
            return Some(Binding {
                reg,
                kind: BindingKind::Int,
                depends_on_stack: true,
            });
        }

        match expr {
            Expr::Lit(ExprLit {
                lit: Lit::Int(int_lit),
                ..
            }) => {
                let value = int_lit.base10_parse::<i64>().ok()?;
                let reg = self.alloc_reg();
                self.statements.push(quote! {
                    __builder.load_const_i_value(#reg, #value);
                });
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: false,
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
                Some(binding)
            }
            Expr::Paren(ExprParen { expr, .. }) => self.lower_value_expr(expr),
            Expr::If(expr_if) => self.lower_if_value(expr_if),
            Expr::Match(expr_match) => self.lower_match_value(expr_match),
            Expr::Unary(ExprUnary { op, expr, .. }) => self.lower_unary(op, expr),
            Expr::Binary(binary) => self.lower_binary(binary),
            Expr::Call(call) => self.lower_call_value(call),
            _ => None,
        }
    }

    fn lower_call_value(&mut self, call: &ExprCall) -> Option<Binding> {
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
        match policy {
            CallPolicySpec::Explicit(kind) => match kind.to_string().as_str() {
                "residual_int" => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_int(__fn_idx, &[#(#arg_regs),*], #reg);
                        });
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_int_typed(__fn_idx, #typed_args, #reg);
                        });
                    }
                }
                "may_force_int" => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_may_force_int(__fn_idx, &[#(#arg_regs),*], #reg);
                        });
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_may_force_int_typed(__fn_idx, #typed_args, #reg);
                        });
                    }
                }
                "release_gil_int" => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_release_gil_int(__fn_idx, &[#(#arg_regs),*], #reg);
                        });
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_release_gil_int_typed(__fn_idx, #typed_args, #reg);
                        });
                    }
                }
                "loopinvariant_int" => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_loopinvariant_int(__fn_idx, &[#(#arg_regs),*], #reg);
                        });
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_loopinvariant_int_typed(__fn_idx, #typed_args, #reg);
                        });
                    }
                }
                "elidable_int" => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_pure_int(__fn_idx, &[#(#arg_regs),*], #reg);
                        });
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        self.statements.push(quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.call_pure_int_typed(__fn_idx, #typed_args, #reg);
                        });
                    }
                }
                "residual_int_wrapped"
                | "may_force_int_wrapped"
                | "release_gil_int_wrapped"
                | "loopinvariant_int_wrapped"
                | "elidable_int_wrapped"
                | "residual_ref_wrapped"
                | "may_force_ref_wrapped"
                | "release_gil_ref_wrapped"
                | "loopinvariant_ref_wrapped"
                | "elidable_ref_wrapped"
                | "residual_float_wrapped"
                | "may_force_float_wrapped"
                | "release_gil_float_wrapped"
                | "loopinvariant_float_wrapped"
                | "elidable_float_wrapped" => {
                    let policy_path = helper_policy_path(&call.func)?;
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let call_stmt = match kind.to_string().as_str() {
                        "residual_int_wrapped" => {
                            quote! { __builder.call_int_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "may_force_int_wrapped" => {
                            quote! { __builder.call_may_force_int_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "release_gil_int_wrapped" => {
                            quote! { __builder.call_release_gil_int_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "loopinvariant_int_wrapped" => {
                            quote! { __builder.call_loopinvariant_int_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "elidable_int_wrapped" => {
                            quote! { __builder.call_pure_int_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "residual_ref_wrapped" => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_ref_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "may_force_ref_wrapped" => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_may_force_ref_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "release_gil_ref_wrapped" => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_release_gil_ref_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "loopinvariant_ref_wrapped" => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_loopinvariant_ref_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "elidable_ref_wrapped" => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_pure_ref_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "residual_float_wrapped" => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_float_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "may_force_float_wrapped" => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_may_force_float_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "release_gil_float_wrapped" => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_release_gil_float_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "loopinvariant_float_wrapped" => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_loopinvariant_float_typed(__fn_idx, #typed_args, #reg); }
                        }
                        "elidable_float_wrapped" => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_pure_float_typed(__fn_idx, #typed_args, #reg); }
                        }
                        _ => unreachable!(),
                    };
                    self.statements.push(quote! {
                        let (__policy, _inline_builder, __trace_target, __concrete_target) = #policy_path();
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
                        let __fn_idx = __builder.add_call_target(__trace_target, __concrete_target);
                        #call_stmt
                    });
                }
                "inline_int" => {
                    let builder_path = inline_builder_path(&call.func)?;
                    let inline_args = typed_inline_arg_tokens(&arg_bindings);
                    self.statements.push(quote! {
                        let (__sub_jitcode, __sub_return_reg, __sub_return_kind) = #builder_path();
                        let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
                        __builder.inline_call_with_typed_args(
                            __sub_idx,
                            #inline_args,
                            Some((__sub_return_reg, #reg)),
                            __sub_return_kind,
                        );
                    });
                }
                _ => return None,
            },
            CallPolicySpec::Infer => {
                let policy_path = helper_policy_path(&call.func)?;
                let typed_args = typed_call_arg_tokens(&arg_bindings);
                let inline_args = typed_inline_arg_tokens(&arg_bindings);
                let int_arg_regs = int_arg_regs(&arg_bindings);
                let unsupported = self.inference_failure_tokens(
                    "inferred helper policy does not support this value call here",
                );
                if let Some(_arg_regs) = int_arg_regs {
                    self.statements.push(quote! {
                        let (__policy, __inline_builder, __trace_target, __concrete_target) = #policy_path();
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
                        let __fn_idx = __builder.add_call_target(__trace_target, __concrete_target);
                        match __policy {
                            2u8 => {
                                __builder.call_int_typed(__fn_idx, #typed_args, #reg);
                            }
                            3u8 => {
                                __builder.call_pure_int_typed(__fn_idx, #typed_args, #reg);
                            }
                            4u8 => {
                                let __builder_fn: fn() -> (majit_meta::JitCode, u16, u8) =
                                    unsafe { std::mem::transmute(__inline_builder) };
                                let (__sub_jitcode, __sub_return_reg, __sub_return_kind) = __builder_fn();
                                let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
                                __builder.inline_call_with_typed_args(
                                    __sub_idx,
                                    #inline_args,
                                    Some((__sub_return_reg, #reg)),
                                    __sub_return_kind,
                                );
                            }
                            10u8 => {
                                __builder.call_may_force_int_typed(__fn_idx, #typed_args, #reg);
                            }
                            14u8 => {
                                __builder.call_release_gil_int_typed(__fn_idx, #typed_args, #reg);
                            }
                            18u8 => {
                                __builder.call_loopinvariant_int_typed(__fn_idx, #typed_args, #reg);
                            }
                            _ => {
                                #unsupported
                            }
                        }
                    });
                } else {
                    self.statements.push(quote! {
                        let (__policy, __inline_builder, __trace_target, __concrete_target) = #policy_path();
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
                        let __fn_idx = __builder.add_call_target(__trace_target, __concrete_target);
                        match __policy {
                            2u8 => {
                                __builder.call_int_typed(__fn_idx, #typed_args, #reg);
                            }
                            3u8 => {
                                __builder.call_pure_int_typed(__fn_idx, #typed_args, #reg);
                            }
                            4u8 => {
                                let __builder_fn: fn() -> (majit_meta::JitCode, u16, u8) =
                                    unsafe { std::mem::transmute(__inline_builder) };
                                let (__sub_jitcode, __sub_return_reg, __sub_return_kind) = __builder_fn();
                                let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
                                __builder.inline_call_with_typed_args(
                                    __sub_idx,
                                    #inline_args,
                                    Some((__sub_return_reg, #reg)),
                                    __sub_return_kind,
                                );
                            }
                            10u8 => {
                                __builder.call_may_force_int_typed(__fn_idx, #typed_args, #reg);
                            }
                            14u8 => {
                                __builder.call_release_gil_int_typed(__fn_idx, #typed_args, #reg);
                            }
                            18u8 => {
                                __builder.call_loopinvariant_int_typed(__fn_idx, #typed_args, #reg);
                            }
                            _ => {
                                #unsupported
                            }
                        }
                    });
                }
            }
        }

        Some(Binding {
            reg,
            kind: result_kind,
            depends_on_stack,
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
        let else_label = self.alloc_label();
        let end_label = self.alloc_label();
        let result_reg = self.alloc_reg();
        let cond_reg = cond.reg;
        let (then_stmts, then_binding) =
            self.lower_branch_value_expr(&Expr::Block(syn::ExprBlock {
                attrs: Vec::new(),
                label: None,
                block: expr_if.then_branch.clone(),
            }))?;
        let (else_stmts, else_binding) = self.lower_branch_value_expr(else_expr)?;
        if !matches!(then_binding.kind, BindingKind::Int)
            || !matches!(else_binding.kind, BindingKind::Int)
        {
            return None;
        }
        let then_reg = then_binding.reg;
        let else_reg = else_binding.reg;

        self.statements.push(quote! {
            let #else_label = __builder.new_label();
        });
        self.statements.push(quote! {
            let #end_label = __builder.new_label();
        });
        self.statements.push(quote! {
            __builder.branch_reg_zero(#cond_reg, #else_label);
        });
        self.statements.extend(then_stmts);
        self.statements.push(quote! {
            __builder.move_i(#result_reg, #then_reg);
        });
        self.statements.push(quote! {
            __builder.jump(#end_label);
        });
        self.statements.push(quote! {
            __builder.mark_label(#else_label);
        });
        self.statements.extend(else_stmts);
        self.statements.push(quote! {
            __builder.move_i(#result_reg, #else_reg);
        });
        self.statements.push(quote! {
            __builder.mark_label(#end_label);
        });

        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack: cond.depends_on_stack
                || then_binding.depends_on_stack
                || else_binding.depends_on_stack,
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
                self.statements.push(quote! {
                    __builder.load_const_i_value(#zero_reg, 0);
                });
                let reg = self.alloc_reg();
                let cond_reg = cond.reg;
                self.statements.push(quote! {
                    __builder.record_binop_i(#reg, majit_ir::OpCode::IntEq, #cond_reg, #zero_reg);
                });
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: cond.depends_on_stack,
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
                self.statements.push(quote! {
                    __builder.record_unary_i(#reg, majit_ir::OpCode::IntNeg, #src_reg);
                });
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: inner.depends_on_stack,
                })
            }
            _ => None,
        }
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
        self.statements.push(quote! {
            __builder.record_binop_i(#reg, majit_ir::OpCode::#opcode, #lhs_reg, #rhs_reg);
        });
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: lhs.depends_on_stack || rhs.depends_on_stack,
        })
    }

    fn lower_branch_expr(&mut self, expr: &Expr) -> Option<Vec<TokenStream>> {
        let stmts = extract_stmts(expr);
        let mut nested = Lowerer {
            bindings: self.bindings.clone(),
            statements: Vec::new(),
            next_reg: self.next_reg,
            next_label: self.next_label,
            config: self.config,
            call_policies: self.call_policies.clone(),
            inference_failure_mode: self.inference_failure_mode,
            auto_calls: self.auto_calls,
        };

        for stmt in &stmts {
            nested.lower_stmt(stmt)?;
        }

        self.next_reg = self.next_reg.max(nested.next_reg);
        self.next_label = self.next_label.max(nested.next_label);
        Some(nested.statements)
    }

    fn lower_branch_value_expr(&mut self, expr: &Expr) -> Option<(Vec<TokenStream>, Binding)> {
        let mut nested = Lowerer {
            bindings: self.bindings.clone(),
            statements: Vec::new(),
            next_reg: self.next_reg,
            next_label: self.next_label,
            config: self.config,
            call_policies: self.call_policies.clone(),
            inference_failure_mode: self.inference_failure_mode,
            auto_calls: self.auto_calls,
        };

        let binding = nested.lower_scoped_value_expr(expr)?;
        self.next_reg = self.next_reg.max(nested.next_reg);
        self.next_label = self.next_label.max(nested.next_label);
        Some((nested.statements, binding))
    }

    fn lower_scoped_value_expr(&mut self, expr: &Expr) -> Option<Binding> {
        match expr {
            Expr::Block(block) => self.lower_block_value(&block.block),
            _ => self.lower_value_expr(expr),
        }
    }

    fn lower_block_value(&mut self, block: &Block) -> Option<Binding> {
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

// ── Loop control detection ───────────────────────────────────────────

/// Check if a block contains break or continue at the top level (not nested in inner loops).
fn block_has_loop_control(block: &Block) -> bool {
    block.stmts.iter().any(|stmt| stmt_has_loop_control(stmt))
}

fn stmt_has_loop_control(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Expr(expr, _) => expr_has_loop_control(expr),
        _ => false,
    }
}

fn expr_has_loop_control(expr: &Expr) -> bool {
    match expr {
        Expr::Break(_) | Expr::Continue(_) => true,
        Expr::If(expr_if) => {
            block_has_loop_control(&expr_if.then_branch)
                || expr_if
                    .else_branch
                    .as_ref()
                    .is_some_and(|(_, e)| expr_has_loop_control(e))
        }
        Expr::Block(block) => block_has_loop_control(&block.block),
        // Don't recurse into nested loops — they have their own break/continue scope
        Expr::Loop(_) | Expr::While(_) | Expr::ForLoop(_) => false,
        _ => false,
    }
}

// ── Helper functions ─────────────────────────────────────────────────

/// Extract the get_mut argument from a pool.get_mut(arg) expression.
fn extract_pool_get_mut_arg(expr: &Expr, config: &LowererConfig) -> Option<Expr> {
    let Expr::MethodCall(mc) = expr else {
        return None;
    };
    if mc.method != "get_mut" || mc.args.len() != 1 {
        return None;
    }
    let receiver_str = normalize_expr(&mc.receiver);
    if receiver_str != config.pool_str {
        return None;
    }
    Some(mc.args[0].clone())
}

fn extract_stmts(expr: &Expr) -> Vec<Stmt> {
    match expr {
        Expr::Block(block) => block.block.stmts.clone(),
        _ => vec![Stmt::Expr(expr.clone(), None)],
    }
}

/// Extract integer literal values from a match arm pattern.
///
/// Supports `Pat::Lit` (integer literals), `Pat::Or` (multiple patterns
/// like `1 | 2 | 3`), and `Pat::Path` (constant paths — evaluated at
/// compile time via `#pat as i64`).
///
/// Returns `None` if the pattern contains unsupported constructs.
fn extract_pat_literals(pat: &Pat) -> Option<Vec<i64>> {
    match pat {
        Pat::Lit(expr_lit) => {
            if let Lit::Int(int_lit) = &expr_lit.lit {
                Some(vec![int_lit.base10_parse::<i64>().ok()?])
            } else {
                None
            }
        }
        Pat::Or(pat_or) => {
            let mut values = Vec::new();
            for case in &pat_or.cases {
                values.extend(extract_pat_literals(case)?);
            }
            Some(values)
        }
        // Constant path pattern (e.g., `MY_CONST`): we cannot evaluate
        // this at proc-macro time, so return None to bail out.
        _ => None,
    }
}

fn push_call_arg(expr: &Expr) -> Option<&Expr> {
    let Expr::MethodCall(ExprMethodCall { method, args, .. }) = expr else {
        return None;
    };
    if matches!(
        method.to_string().as_str(),
        "push" | "push_ref" | "push_float"
    ) && args.len() == 1
    {
        Some(&args[0])
    } else {
        None
    }
}

fn int_arg_regs(bindings: &[Binding]) -> Option<Vec<u16>> {
    bindings
        .iter()
        .map(|binding| match binding.kind {
            BindingKind::Int => Some(binding.reg),
            BindingKind::Ref | BindingKind::Float => None,
        })
        .collect()
}

fn typed_inline_arg_tokens(bindings: &[Binding]) -> TokenStream {
    let args = bindings.iter().enumerate().map(|(index, binding)| {
        let reg = binding.reg;
        let idx = index as u16;
        match binding.kind {
            BindingKind::Int => {
                quote! { (majit_meta::JitArgKind::Int, #reg, #idx) }
            }
            BindingKind::Ref => {
                quote! { (majit_meta::JitArgKind::Ref, #reg, #idx) }
            }
            BindingKind::Float => {
                quote! { (majit_meta::JitArgKind::Float, #reg, #idx) }
            }
        }
    });
    quote! { &[#(#args),*] }
}

fn typed_call_arg_tokens(bindings: &[Binding]) -> TokenStream {
    let args = bindings.iter().map(|binding| {
        let reg = binding.reg;
        match binding.kind {
            BindingKind::Int => quote! { majit_meta::JitCallArg::int(#reg) },
            BindingKind::Ref => quote! { majit_meta::JitCallArg::reference(#reg) },
            BindingKind::Float => quote! { majit_meta::JitCallArg::float(#reg) },
        }
    });
    quote! { &[#(#args),*] }
}

fn is_pop_call(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::MethodCall(ExprMethodCall { method, args, .. }) if method == "pop" && args.is_empty()
    )
}

fn is_pop_value(expr: &Expr) -> bool {
    if is_pop_call(expr) {
        return true;
    }
    match expr {
        Expr::MethodCall(ExprMethodCall {
            receiver,
            method,
            args,
            ..
        }) if (method == "unwrap" || method == "expect")
            && (method == "unwrap" || args.len() == 1) =>
        {
            is_pop_call(receiver)
        }
        Expr::Call(ExprCall { func, args, .. }) if args.is_empty() => {
            matches!(&**func, Expr::MethodCall(ExprMethodCall { receiver, method, .. }) if method == "unwrap" && is_pop_call(receiver))
        }
        _ => false,
    }
}

fn is_peek_value(expr: &Expr) -> bool {
    match expr {
        Expr::MethodCall(ExprMethodCall { method, args, .. })
            if method == "peek" && args.is_empty() =>
        {
            true
        }
        Expr::MethodCall(ExprMethodCall {
            receiver,
            method,
            args,
            ..
        }) if (method == "unwrap" || method == "expect")
            && (method == "unwrap" || args.len() == 1) =>
        {
            matches!(
                &**receiver,
                Expr::MethodCall(ExprMethodCall { method, args, .. })
                    if method == "peek" && args.is_empty()
            )
        }
        _ => false,
    }
}

fn is_pop_discard(expr: &Expr) -> bool {
    is_pop_call(expr)
}

fn is_supported_int_cast(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => type_path.path.is_ident("i64") || type_path.path.is_ident("isize"),
        _ => false,
    }
}

fn is_supported_ref_type(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => type_path.path.is_ident("usize"),
        Type::Ptr(_) => true,
        _ => false,
    }
}

fn is_supported_float_type(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => type_path.path.is_ident("f64"),
        _ => false,
    }
}

pub(crate) fn classify_param_type(ty: &Type) -> Option<InlineReturnKind> {
    if is_supported_int_cast(ty) {
        Some(InlineReturnKind::Int)
    } else if is_supported_ref_type(ty) {
        Some(InlineReturnKind::Ref)
    } else if is_supported_float_type(ty) {
        Some(InlineReturnKind::Float)
    } else {
        None
    }
}

fn extract_bool_branch_values(expr_if: &ExprIf) -> Option<(i64, i64)> {
    let then_value = extract_block_tail_int(&expr_if.then_branch)?;
    let (_, else_expr) = expr_if.else_branch.as_ref()?;
    let else_value = extract_branch_int(else_expr)?;
    Some((then_value, else_value))
}

fn extract_block_tail_int(block: &Block) -> Option<i64> {
    match block.stmts.as_slice() {
        [Stmt::Expr(expr, None)] => extract_branch_int(expr),
        _ => None,
    }
}

fn extract_branch_int(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(int_lit),
            ..
        }) => int_lit.base10_parse::<i64>().ok(),
        Expr::Paren(ExprParen { expr, .. }) => extract_branch_int(expr),
        Expr::Block(block) => extract_block_tail_int(&block.block),
        _ => None,
    }
}

fn inline_builder_path(expr: &Expr) -> Option<Path> {
    let Expr::Path(ExprPath { path, .. }) = expr else {
        return None;
    };
    let mut path = path.clone();
    let last = path.segments.last_mut()?;
    last.ident = format_ident!("__majit_inline_jitcode_{}", last.ident);
    Some(path)
}

fn helper_policy_path(expr: &Expr) -> Option<Path> {
    let Expr::Path(ExprPath { path, .. }) = expr else {
        return None;
    };
    let mut path = path.clone();
    let last = path.segments.last_mut()?;
    last.ident = format_ident!("__majit_call_policy_{}", last.ident);
    Some(path)
}

fn opcode_for_binop(op: &BinOp) -> Option<Ident> {
    let name = match op {
        BinOp::Add(_) => "IntAdd",
        BinOp::Sub(_) => "IntSub",
        BinOp::Mul(_) => "IntMul",
        BinOp::Div(_) => "IntFloorDiv",
        BinOp::Rem(_) => "IntMod",
        BinOp::BitAnd(_) => "IntAnd",
        BinOp::BitOr(_) => "IntOr",
        BinOp::BitXor(_) => "IntXor",
        BinOp::Shl(_) => "IntLshift",
        BinOp::Shr(_) => "IntRshift",
        BinOp::Eq(_) => "IntEq",
        BinOp::Ne(_) => "IntNe",
        BinOp::Lt(_) => "IntLt",
        BinOp::Le(_) => "IntLe",
        BinOp::Gt(_) => "IntGt",
        BinOp::Ge(_) => "IntGe",
        _ => return None,
    };
    Some(Ident::new(name, proc_macro2::Span::call_site()))
}

// ── Public entry points ──────────────────────────────────────────────

pub fn try_generate_jitcode_body(body: &Expr) -> Option<TokenStream> {
    try_generate_jitcode_body_inner(body, None)
}

pub fn try_generate_jitcode_body_with_config(
    config: &LowererConfig,
    body: &Expr,
) -> Option<TokenStream> {
    try_generate_jitcode_body_inner(body, Some(config))
}

pub(crate) fn generate_inline_helper_jitcode_with_calls(
    func: &ItemFn,
    calls: &[(Path, Option<Ident>)],
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
        .map(|(path, kind)| {
            let spec = match kind {
                Some(kind) => CallPolicySpec::Explicit(kind.clone()),
                None => CallPolicySpec::Infer,
            };
            (normalize(&quote!(#path).to_string()), spec)
        })
        .collect();
    let mut lowerer =
        Lowerer::new_with_call_policies(None, call_policies, InferenceFailureMode::Panic);
    for (index, arg) in func.sig.inputs.iter().enumerate() {
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
        lowerer.bindings.insert(
            pat_ident.ident.to_string(),
            Binding {
                reg: index as u16,
                kind: binding_kind,
                depends_on_stack: false,
            },
        );
    }
    lowerer.next_reg = func.sig.inputs.len() as u16;

    let Some(binding) = lowerer.lower_block_value(&func.block) else {
        return Ok(None);
    };

    let statements = lowerer.statements;
    Ok(Some(InlineHelperJitCode {
        body: quote! {
            #(#statements)*
        },
        return_reg: binding.reg,
        return_kind,
    }))
}

fn try_generate_jitcode_body_inner(
    body: &Expr,
    config: Option<&LowererConfig>,
) -> Option<TokenStream> {
    let stmts = extract_stmts(body);
    if stmts.is_empty() {
        return None;
    }

    let mut lowerer = Lowerer::new(config);
    for stmt in &stmts {
        lowerer.lower_stmt(stmt)?;
    }

    let statements = lowerer.statements;
    Some(quote! {
        #(#statements)*
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_lower(code: &str) -> Option<String> {
        let expr: Expr = syn::parse_str(code).expect("failed to parse");
        let result = try_generate_jitcode_body(&expr)?;
        Some(result.to_string())
    }

    #[test]
    fn lower_match_stmt_with_literals() {
        let code = r#"{
            let x = stack.pop();
            match x {
                1 => { stack.push(10); }
                2 => { stack.push(20); }
                _ => { stack.push(0); }
            }
        }"#;
        let result = try_lower(code);
        assert!(result.is_some(), "match should be lowerable");
        let s = result.unwrap();
        // Should contain IntEq comparisons and branch labels
        assert!(s.contains("IntEq"), "should generate equality checks");
        assert!(s.contains("branch_reg_zero"), "should generate branches");
        assert!(s.contains("new_label"), "should generate labels");
    }

    #[test]
    fn lower_match_or_pattern() {
        let code = r#"{
            let x = stack.pop();
            match x {
                1 | 2 => { stack.push(10); }
                _ => { stack.push(0); }
            }
        }"#;
        let result = try_lower(code);
        assert!(
            result.is_some(),
            "match with Or pattern should be lowerable"
        );
        let s = result.unwrap();
        assert!(
            s.contains("IntOr"),
            "should generate OR for multi-literal pattern"
        );
    }

    #[test]
    fn lower_match_value_expr() {
        let code = r#"{
            let x = stack.pop();
            let y = match x {
                1 => 10,
                2 => 20,
                _ => 0
            };
            stack.push(y);
        }"#;
        let result = try_lower(code);
        assert!(result.is_some(), "match as value should be lowerable");
        let s = result.unwrap();
        assert!(
            s.contains("move_i"),
            "should produce move_i for value result"
        );
    }

    fn parse_pat(code: &str) -> Pat {
        let match_code = format!("match x {{ {code} => () }}");
        let expr: syn::ExprMatch = syn::parse_str(&match_code).expect("failed to parse match");
        expr.arms.into_iter().next().unwrap().pat
    }

    #[test]
    fn extract_pat_literals_single() {
        let pat = parse_pat("42");
        let lits = extract_pat_literals(&pat);
        assert_eq!(lits, Some(vec![42]));
    }

    #[test]
    fn extract_pat_literals_or() {
        let pat = parse_pat("1 | 2 | 3");
        let lits = extract_pat_literals(&pat);
        assert_eq!(lits, Some(vec![1, 2, 3]));
    }

    #[test]
    fn extract_pat_literals_wildcard_returns_none() {
        let pat = parse_pat("_");
        let lits = extract_pat_literals(&pat);
        assert_eq!(lits, None);
    }

    #[test]
    fn lower_match_no_default() {
        // Match without a wildcard arm; all arms are guarded.
        let code = r#"{
            let x = stack.pop();
            match x {
                1 => { stack.push(10); }
                2 => { stack.push(20); }
            }
        }"#;
        let result = try_lower(code);
        assert!(
            result.is_some(),
            "match without default should be lowerable"
        );
    }

    #[test]
    fn lower_while_loop() {
        let code = r#"{
            let x = stack.pop();
            while x > 0 {
                stack.push(x);
                break;
            }
        }"#;
        let result = try_lower(code);
        assert!(result.is_some(), "while loop should be lowerable");
        let s = result.unwrap();
        assert!(s.contains("mark_label"), "should generate loop labels");
        assert!(s.contains("branch_reg_zero"), "should generate exit branch");
        assert!(s.contains("jump"), "should generate back-edge jump");
    }

    #[test]
    fn lower_loop_with_break() {
        let code = r#"{
            let x = stack.pop();
            loop {
                stack.push(x);
                break;
            }
        }"#;
        let result = try_lower(code);
        assert!(result.is_some(), "loop with break should be lowerable");
        let s = result.unwrap();
        assert!(s.contains("mark_label"), "should generate loop labels");
        assert!(
            s.contains("jump"),
            "should generate jumps for break and back-edge"
        );
    }

    #[test]
    fn lower_loop_with_conditional_break() {
        let code = r#"{
            let x = stack.pop();
            loop {
                if x > 0 {
                    break;
                }
                stack.push(x);
            }
        }"#;
        let result = try_lower(code);
        assert!(
            result.is_some(),
            "loop with conditional break should be lowerable"
        );
        let s = result.unwrap();
        assert!(s.contains("mark_label"), "should generate labels");
        assert!(
            s.contains("branch_reg_zero"),
            "should generate conditional branch"
        );
    }

    #[test]
    fn lower_for_loop_returns_none() {
        // For loops are not lowered (they fall back to opaque).
        let code = r#"{
            for i in 0..10 {
                stack.push(i);
            }
        }"#;
        let result = try_lower(code);
        assert!(result.is_none(), "for loop should not be lowerable");
    }
}
