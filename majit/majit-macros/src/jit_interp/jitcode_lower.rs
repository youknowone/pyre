use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    BinOp, Block, Expr, ExprAssign, ExprBinary, ExprCall, ExprCast, ExprIf, ExprLit,
    ExprMethodCall, ExprParen, ExprPath, ExprUnary, FnArg, Ident, ItemFn, Lit, Local, Pat, Path,
    ReturnType, Stmt, Type, UnOp,
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
}

const MAX_HELPER_CALL_ARITY: usize = 8;

pub(crate) struct InlineHelperJitCode {
    pub body: TokenStream,
    pub param_count: u16,
    pub return_reg: u16,
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
        Self {
            pool_str,
            selector_str,
            pool_get_mut_sel,
            binops,
            io_shims,
            calls,
            auto_calls,
        }
    }
}

fn normalize(s: &str) -> String {
    s.replace(' ', "")
}

fn normalize_expr(expr: &Expr) -> String {
    normalize(&quote!(#expr).to_string())
}

// ── Lowerer ──────────────────────────────────────────────────────────

#[derive(Clone)]
struct Binding {
    reg: u16,
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
                    depends_on_stack: false,
                },
            );
            return Some(());
        }

        None
    }

    fn lower_expr_stmt(&mut self, expr: &Expr) -> Option<()> {
        // Config-aware: push_to (non-current storage) BEFORE regular push
        if self.config.is_some() {
            if let Some(()) = self.lower_push_to_stmt(expr) {
                return Some(());
            }
        }

        if let Some(push_arg) = push_call_arg(expr) {
            let binding = self.lower_value_expr(push_arg)?;
            let reg = binding.reg;
            self.statements.push(quote! {
                __builder.push_i(#reg);
            });
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
            self.statements.push(quote! {
                {
                    let mut __sub = majit_meta::JitCodeBuilder::new();
                    __sub.pop_i(0);
                    __sub.pop_i(1);
                    __sub.record_binop_i(2, majit_ir::OpCode::#opcode, 1, 0);
                    __sub.push_i(2);
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

        let mut arg_regs = Vec::with_capacity(call.args.len());
        for arg in &call.args {
            let binding = self.lower_value_expr(arg)?;
            arg_regs.push(binding.reg);
        }
        let func = &call.func;
        match policy {
            CallPolicySpec::Explicit(kind) => {
                if kind.to_string() != "residual_void" {
                    return None;
                }
                self.statements.push(quote! {
                    let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                    __builder.residual_call_void_args(__fn_idx, &[#(#arg_regs),*]);
                });
            }
            CallPolicySpec::Infer => {
                let policy_path = helper_policy_path(&call.func)?;
                let unsupported = self.inference_failure_tokens(
                    "inferred helper policy does not support void calls here",
                );
                self.statements.push(quote! {
                    let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                    let (__policy, _inline_builder) = #policy_path();
                    match __policy {
                        1u8 => {
                            __builder.residual_call_void_args(__fn_idx, &[#(#arg_regs),*]);
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
                let arg = call.args.first()?;
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

    /// Lower selector assignment: state.selected = value → set_selected(const_idx)
    fn lower_selector_assign(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Assign(ExprAssign { left, right, .. }) = expr else {
            return None;
        };
        let config = self.config?;
        let lhs_str = normalize_expr(left);
        if lhs_str != config.selector_str {
            return None;
        }
        let rhs = &**right;
        self.statements.push(quote! {
            let __sel_const = __builder.add_const_i((#rhs) as i64);
            __builder.set_selected(__sel_const);
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

    fn lower_value_expr(&mut self, expr: &Expr) -> Option<Binding> {
        if is_pop_value(expr) {
            let reg = self.alloc_reg();
            self.statements.push(quote! {
                __builder.pop_i(#reg);
            });
            return Some(Binding {
                reg,
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
                    depends_on_stack: false,
                })
            }
            Expr::Path(ExprPath { path, .. }) => {
                let ident = path.get_ident()?;
                self.bindings.get(&ident.to_string()).cloned()
            }
            Expr::Cast(ExprCast { expr, ty, .. }) if is_supported_int_cast(ty) => {
                self.lower_value_expr(expr)
            }
            Expr::Paren(ExprParen { expr, .. }) => self.lower_value_expr(expr),
            Expr::If(expr_if) => self.lower_if_value(expr_if),
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

        let mut arg_regs = Vec::with_capacity(call.args.len());
        let mut depends_on_stack = false;
        for arg in &call.args {
            let binding = self.lower_value_expr(arg)?;
            arg_regs.push(binding.reg);
            depends_on_stack |= binding.depends_on_stack;
        }

        let reg = self.alloc_reg();
        let func = &call.func;
        match policy {
            CallPolicySpec::Explicit(kind) => match kind.to_string().as_str() {
                "residual_int" => {
                    self.statements.push(quote! {
                        let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                        __builder.call_int(__fn_idx, &[#(#arg_regs),*], #reg);
                    });
                }
                "elidable_int" => {
                    self.statements.push(quote! {
                        let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                        __builder.call_pure_int(__fn_idx, &[#(#arg_regs),*], #reg);
                    });
                }
                "inline_int" => {
                    let builder_path = inline_builder_path(&call.func)?;
                    let callee_arg_map = arg_regs
                        .iter()
                        .enumerate()
                        .map(|(index, caller_reg)| quote! { (#caller_reg, #index as u16) });
                    self.statements.push(quote! {
                        let (__sub_jitcode, __sub_return_reg) = #builder_path();
                        let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
                        __builder.inline_call_i(
                            __sub_idx,
                            &[#(#callee_arg_map),*],
                            Some((__sub_return_reg, #reg)),
                        );
                    });
                }
                _ => return None,
            },
            CallPolicySpec::Infer => {
                let policy_path = helper_policy_path(&call.func)?;
                let unsupported = self.inference_failure_tokens(
                    "inferred helper policy does not support integer value calls here",
                );
                let callee_arg_map = arg_regs
                    .iter()
                    .enumerate()
                    .map(|(index, caller_reg)| quote! { (#caller_reg, #index as u16) });
                self.statements.push(quote! {
                    let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                    let (__policy, __inline_builder) = #policy_path();
                    match __policy {
                        2u8 => {
                            __builder.call_int(__fn_idx, &[#(#arg_regs),*], #reg);
                        }
                        3u8 => {
                            __builder.call_pure_int(__fn_idx, &[#(#arg_regs),*], #reg);
                        }
                        4u8 => {
                            let __builder_fn: fn() -> (majit_meta::JitCode, u16) =
                                unsafe { std::mem::transmute(__inline_builder) };
                            let (__sub_jitcode, __sub_return_reg) = __builder_fn();
                            let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
                            __builder.inline_call_i(
                                __sub_idx,
                                &[#(#callee_arg_map),*],
                                Some((__sub_return_reg, #reg)),
                            );
                        }
                        _ => {
                            #unsupported
                        }
                    }
                });
            }
        }

        Some(Binding {
            reg,
            depends_on_stack,
        })
    }

    fn lower_if_value(&mut self, expr_if: &ExprIf) -> Option<Binding> {
        if let Some(binding) = self.lower_bool_if(expr_if) {
            return Some(binding);
        }

        let cond = self.lower_value_expr(&expr_if.cond)?;
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
            depends_on_stack: cond.depends_on_stack
                || then_binding.depends_on_stack
                || else_binding.depends_on_stack,
        })
    }

    fn lower_bool_if(&mut self, expr_if: &ExprIf) -> Option<Binding> {
        let (then_value, else_value) = extract_bool_branch_values(expr_if)?;
        let cond = self.lower_value_expr(&expr_if.cond)?;
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
                let reg = self.alloc_reg();
                let src_reg = inner.reg;
                self.statements.push(quote! {
                    __builder.record_unary_i(#reg, majit_ir::OpCode::IntNeg, #src_reg);
                });
                Some(Binding {
                    reg,
                    depends_on_stack: inner.depends_on_stack,
                })
            }
            _ => None,
        }
    }

    fn lower_binary(&mut self, expr: &ExprBinary) -> Option<Binding> {
        let lhs = self.lower_value_expr(&expr.left)?;
        let rhs = self.lower_value_expr(&expr.right)?;
        let opcode = opcode_for_binop(&expr.op)?;
        let reg = self.alloc_reg();
        let lhs_reg = lhs.reg;
        let rhs_reg = rhs.reg;
        self.statements.push(quote! {
            __builder.record_binop_i(#reg, majit_ir::OpCode::#opcode, #lhs_reg, #rhs_reg);
        });
        Some(Binding {
            reg,
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

fn push_call_arg(expr: &Expr) -> Option<&Expr> {
    let Expr::MethodCall(ExprMethodCall { method, args, .. }) = expr else {
        return None;
    };
    if method == "push" && args.len() == 1 {
        Some(&args[0])
    } else {
        None
    }
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
            "#[jit_inline] requires an integer return type",
        ));
    };
    if !is_supported_int_cast(return_ty) {
        return Err(syn::Error::new_spanned(
            return_ty,
            "#[jit_inline] currently supports only i64/isize return types",
        ));
    }

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
        if !is_supported_int_cast(&pat_type.ty) {
            return Err(syn::Error::new_spanned(
                &pat_type.ty,
                "#[jit_inline] parameters must use i64/isize",
            ));
        }
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(syn::Error::new_spanned(
                &pat_type.pat,
                "#[jit_inline] parameters must be simple identifiers",
            ));
        };
        lowerer.bindings.insert(
            pat_ident.ident.to_string(),
            Binding {
                reg: index as u16,
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
        param_count: func.sig.inputs.len() as u16,
        return_reg: binding.reg,
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
