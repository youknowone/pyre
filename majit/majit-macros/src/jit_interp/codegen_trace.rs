//! Generate `JitCode` builders and the generic `__trace_*` wrapper.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ItemFn;

use super::JitInterpConfig;
use super::classify::{ArmPattern, classify_arms};
use super::jitcode_lower::{self, LowererConfig};

pub fn generate_trace_fn(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    let fn_name = &func.sig.ident;
    let trace_fn_name = format_ident!("__trace_{}", fn_name);
    let jitcode_fn_name = format_ident!("__jitcode_{}", fn_name);

    let match_expr = find_dispatch_match(&func.block);
    let Some(match_expr) = match_expr else {
        return syn::Error::new_spanned(func, "could not find opcode dispatch match")
            .to_compile_error();
    };

    let lowerer_config = LowererConfig::new(
        config.storage.as_ref().map(|s| &s.pool),
        config.storage.as_ref().map(|s| &s.selector),
        &config.binops,
        &config.io_shims,
        &config.calls,
        config.auto_calls,
        config.virtualizable_decl.as_ref(),
        config.state_fields.as_ref(),
    );

    let classified = classify_arms(&match_expr.arms);
    let env_type = &config.env_type;
    let has_branch_group = classified
        .iter()
        .any(|arm| matches!(arm.pattern, ArmPattern::BranchGroup));

    let jitcode_arms = classified
        .iter()
        .map(|arm| generate_jitcode_arm(arm, &lowerer_config));

    let label_closure = if has_branch_group {
        quote! { |__jit_pc| program.get_label(__jit_pc) }
    } else {
        quote! { |_unused_pc| 0usize }
    };

    let trace_fn_body = if config.state_fields.is_some() {
        // state_fields mode: no storage pool, use dummy closures.
        // State field operations use load_state_field/store_state_field JitCode ops
        // which don't call the runtime stack closures.
        quote! {
            #[allow(non_snake_case, unused_variables, unused_mut)]
            fn #trace_fn_name(
                __ctx: &mut majit_meta::TraceCtx,
                __sym: &mut __JitSym,
                program: &#env_type,
                pc: usize,
            ) -> majit_meta::TraceAction {
                use majit_meta::TraceAction;

                let __op = program.get_op(pc);
                let Some(__jitcode) = #jitcode_fn_name(program, pc, __op) else {
                    if majit_meta::majit_log_enabled() {
                        eprintln!(
                            "[jit] no jitcode for pc={} op={}",
                            pc,
                            __op
                        );
                    }
                    return TraceAction::AbortPermanent;
                };

                let __result = majit_meta::trace_jitcode(
                    __ctx,
                    __sym,
                    &__jitcode,
                    pc,
                    |_| 0usize,
                    |_, _| 0i64,
                    #label_closure,
                );
                if majit_meta::majit_log_enabled() && !matches!(__result, TraceAction::Continue) {
                    eprintln!(
                        "[jit] trace action at pc={} op={} -> {:?}",
                        pc,
                        __op,
                        __result
                    );
                }
                __result
            }
        }
    } else {
        let pool_type = &config
            .storage
            .as_ref()
            .expect("storage config required in storage mode")
            .pool_type;
        quote! {
            #[allow(non_snake_case, unused_variables, unused_mut)]
            fn #trace_fn_name(
                __ctx: &mut majit_meta::TraceCtx,
                __sym: &mut __JitSym,
                program: &#env_type,
                pc: usize,
                __storage: &#pool_type,
                __selected: usize,
            ) -> majit_meta::TraceAction {
                use majit_meta::TraceAction;

                let __op = program.get_op(pc);
                let Some(__jitcode) = #jitcode_fn_name(program, pc, __op) else {
                    if majit_meta::majit_log_enabled() {
                        eprintln!(
                            "[jit] no jitcode for pc={} op={} selected={}",
                            pc,
                            __op,
                            __selected
                        );
                    }
                    return TraceAction::AbortPermanent;
                };

                let __result = majit_meta::trace_jitcode(
                    __ctx,
                    __sym,
                    &__jitcode,
                    pc,
                    |__stack_index| __storage.get(__stack_index).len(),
                    |__stack_index, __pos| __storage.get(__stack_index).peek_at(__pos),
                    #label_closure,
                );
                if majit_meta::majit_log_enabled() && !matches!(__result, TraceAction::Continue) {
                    eprintln!(
                        "[jit] trace action at pc={} op={} selected={} -> {:?}",
                        pc,
                        __op,
                        __selected,
                        __result
                    );
                }
                __result
            }
        }
    };

    quote! {
        #[allow(non_snake_case, unused_variables, unused_mut)]
        fn #jitcode_fn_name(
            program: &#env_type,
            pc: usize,
            __op: u8,
        ) -> Option<majit_meta::JitCode> {
            match __op {
                #(#jitcode_arms)*
            }
        }

        #trace_fn_body
    }
}

fn generate_jitcode_arm(
    arm: &super::classify::ClassifiedArm,
    config: &LowererConfig,
) -> TokenStream {
    let pat = &arm.pat;
    let build = match &arm.pattern {
        ArmPattern::Lowerable => {
            // Try config-aware lowering first, fall back to basic lowering
            let code =
                jitcode_lower::try_generate_jitcode_body_with_config(config, &arm.original_body)
                    .or_else(|| jitcode_lower::try_generate_jitcode_body(&arm.original_body));

            match code {
                Some(code) => quote! {
                    let mut __builder = majit_meta::JitCodeBuilder::new();
                    #code
                    Some(__builder.finish())
                },
                None => quote! { None },
            }
        }
        ArmPattern::BranchGroup => quote! {
            let mut __builder = majit_meta::JitCodeBuilder::new();
            match __op {
                crate::aheui::OP_BRPOP1 => __builder.require_stack(1),
                crate::aheui::OP_BRPOP2 => __builder.require_stack(2),
                crate::aheui::OP_BRZ => __builder.branch_zero(),
                crate::aheui::OP_JMP => __builder.jump_target(),
                _ => return None,
            }
            Some(__builder.finish())
        },
        ArmPattern::AbortPermanent | ArmPattern::Halt => quote! {
            let mut __builder = majit_meta::JitCodeBuilder::new();
            __builder.abort_permanent();
            Some(__builder.finish())
        },
        ArmPattern::Nop => quote! {
            Some(majit_meta::JitCodeBuilder::new().finish())
        },
        ArmPattern::Unsupported(_reason) => {
            // Complex CFG (loop/while/for in match arm) cannot be lowered to
            // JitCode. Instead of compile_error!, emit an abort bytecode so
            // tracing falls back to the interpreter — matching RPython's
            // dont_look_inside behavior for complex code patterns.
            quote! {
                Some({
                    let mut builder = majit_meta::JitCodeBuilder::new();
                    builder.abort();
                    builder.finish()
                })
            }
        }
    };

    quote! { #pat => { #build }, }
}

fn find_dispatch_match(block: &syn::Block) -> Option<&syn::ExprMatch> {
    for stmt in &block.stmts {
        if let Some(m) = find_match_in_stmt(stmt) {
            return Some(m);
        }
    }
    None
}

fn find_match_in_stmt(stmt: &syn::Stmt) -> Option<&syn::ExprMatch> {
    match stmt {
        syn::Stmt::Expr(expr, _) => find_match_in_expr(expr),
        syn::Stmt::Local(local) => {
            if let Some(init) = &local.init {
                find_match_in_expr(&init.expr)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn find_match_in_expr(expr: &syn::Expr) -> Option<&syn::ExprMatch> {
    match expr {
        syn::Expr::Match(m) => Some(m),
        syn::Expr::While(w) => {
            for stmt in &w.body.stmts {
                if let Some(m) = find_match_in_stmt(stmt) {
                    return Some(m);
                }
            }
            None
        }
        syn::Expr::Loop(l) => {
            for stmt in &l.body.stmts {
                if let Some(m) = find_match_in_stmt(stmt) {
                    return Some(m);
                }
            }
            None
        }
        syn::Expr::Block(b) => {
            for stmt in &b.block.stmts {
                if let Some(m) = find_match_in_stmt(stmt) {
                    return Some(m);
                }
            }
            None
        }
        syn::Expr::If(i) => {
            for stmt in &i.then_branch.stmts {
                if let Some(m) = find_match_in_stmt(stmt) {
                    return Some(m);
                }
            }
            if let Some((_, else_expr)) = &i.else_branch {
                return find_match_in_expr(else_expr);
            }
            None
        }
        _ => None,
    }
}
