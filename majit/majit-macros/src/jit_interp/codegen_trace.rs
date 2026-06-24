//! Generate `JitCode` builders and the generic `__trace_*` wrapper.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Block, Expr, ExprMatch, ItemFn, Stmt};

use super::JitInterpConfig;
use super::classify::classify_arms;
use super::jitcode_lower::{self, LowererConfig};

pub fn generate_trace_fn(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    let fn_name = &func.sig.ident;
    let trace_fn_name = format_ident!("__trace_{}", fn_name);
    let prebuild_fn_name = format_ident!("__prebuild_jitcode_liveness_{}", fn_name);
    let dispatch_jitcode_fn_name = format_ident!("__dispatch_jitcode_{}", fn_name);
    let declare_schema_fn_name = format_ident!("__declare_jit_schema_{}", fn_name);

    let match_expr = find_dispatch_match(&func.block);
    let Some(match_expr) = match_expr else {
        return syn::Error::new_spanned(func, "could not find opcode dispatch match")
            .to_compile_error();
    };

    let lowerer_config = LowererConfig::new(
        &config.io_shims,
        &config.calls,
        config.auto_calls,
        config.virtualizable_decl.as_ref(),
        config.state_fields.as_ref(),
        &config.greens,
        &config.green_type_tags,
        &config.reds,
        &config.state_type,
        &config.env_type,
        &config.residual_writes,
        &config.pool_arrays,
    );

    let classified = classify_arms(&match_expr.arms);
    let env_type = &config.env_type;

    // Dispatch JitCode singleton produced by lower_dispatch_body.
    // `__trace_*` invokes it; the install pipeline registers it as the
    // driver-shared singleton. The prebuild step splices the
    // dispatch JitCode's per-marker liveness prebuild into
    // `__prebuild_jitcode_liveness_*` alongside the per-arm prebuilds, so the
    // driver-shared `Assembler` already holds every triple the dispatch
    // factory will emit — preserving the no-growth invariant asserted in
    // `__trace_*` below for the dispatch JitCode build path.
    let dispatch_lowerer_config = lowerer_config.with_vable_input_ref_reg(1);
    let (
        dispatch_body,
        dispatch_prebuild,
        dispatch_lower_ok,
        dispatch_green_schema,
        dispatch_red_schema,
    ) = match jitcode_lower::lower_dispatch_body(&dispatch_lowerer_config, &func.block, &classified)
    {
        Some(generated) => (
            generated.body,
            generated.liveness_prebuild,
            true,
            generated.green_schema,
            generated.red_schema,
        ),
        None => (quote! {}, quote! {}, false, Vec::new(), Vec::new()),
    };
    // Slice (audit Issue #5) — split the (name, type) tuples into
    // separate name + type token vectors so the macro splat below
    // can interleave them as `(#name, #type)` without losing
    // ordering.  Per-pair iteration via tuple destructuring inside
    // the splat is not supported by `quote!`.
    let dispatch_green_schema_names: Vec<&str> = dispatch_green_schema
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    let dispatch_green_schema_types: Vec<&proc_macro2::TokenStream> =
        dispatch_green_schema.iter().map(|(_, t)| t).collect();
    let dispatch_red_schema_names: Vec<&str> = dispatch_red_schema
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    let dispatch_red_schema_types: Vec<&proc_macro2::TokenStream> =
        dispatch_red_schema.iter().map(|(_, t)| t).collect();

    // Slice X-D production wire-up: the identity `label_at` closure
    // and the `jitcell_token_arc_for_number` resolver are constructed
    // by `generate_merge_wrapper` and passed into `__trace_*` as a
    // `ClosureRuntimeWithResolver`.  Both closures need to live at the
    // merge wrapper layer because the resolver borrows
    // `MetaInterp::compiled_loops` / `warm_state` via the
    // `with_trace_ctx_and_token_resolver` split-borrow helper.
    // (Legacy identity closure `|pc| pc` retained at the
    // call site there.)
    // State-field `[int; virt]` arrays also seed a standard-virtualizable
    // identity: the OpRef of `virtualizable_boxes[-1]` must reach ref reg 1
    // so the `getarrayitem_vable_*` dispatch ops resolve the same box that
    // `init_virtualizable_boxes` minted (identity-equal per
    // `is_nonstandard_virtualizable` Step 3). Push the identity argbox for
    // both the PyFrame-style declaration and the state-field virt-array case.
    let state_has_virt_array = config
        .state_fields
        .as_ref()
        .map(|sf| {
            sf.fields
                .iter()
                .any(|f| matches!(f.kind, crate::jit_interp::StateFieldKind::VirtArray(_)))
        })
        .unwrap_or(false);
    let push_virtualizable_argbox = if config.virtualizable_decl.is_some() || state_has_virt_array {
        quote! {
            let Some(__vable_argbox) = __ctx.standard_virtualizable_jitcode_argbox() else {
                return TraceAction::Abort;
            };
            __jitcode_args.push(__vable_argbox);
        }
    } else {
        quote! {}
    };

    let trace_fn_body = quote! {
        #[allow(non_snake_case, unused_variables, unused_mut)]
        fn #trace_fn_name<__R: majit_metainterp::JitCodeRuntime>(
            __ctx: &mut majit_metainterp::TraceCtx,
            __sym: &mut __JitSym,
            program: &#env_type,
            pc: usize,
            // Slice X-D production wire-up: caller passes a
            // `ClosureRuntimeWithResolver` carrying both `label_at` and
            // the warmstate-backed `jitcell_token_arc_for_number`
            // callback so BC_CALL_ASSEMBLER_* dispatch resolves to the
            // production `Arc<JitCellToken>` instead of the synth-Arc
            // `_by_number_typed` fallback.
            __runtime: &__R,
            // Dispatch JitCode singleton cloned from JitDriver by the
            // caller before the mutable borrow. `None` only when
            // `lower_dispatch_body` returned `None` at proc-macro time
            // (`dispatch_lower_ok == false`) and `register_dispatch_jitcode`
            // was therefore skipped at install — trace aborts permanently
            // in that case, matching `pypy/module/pypyjit/interp_jit.py:82-94`
            // dispatch() which only invokes the singleton dispatch JitCode.
            __dispatch_jitcode_arg: Option<&::std::sync::Arc<majit_metainterp::JitCode>>,
        ) -> majit_metainterp::TraceAction {
            use majit_metainterp::TraceAction;

            let Some(__dispatch_arc) = __dispatch_jitcode_arg else {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit] no dispatch JitCode registered at pc={} \
                         (lower_dispatch_body returned None at proc-macro time)",
                        pc,
                    );
                }
                return TraceAction::AbortPermanent;
            };
            let __jitcode: majit_metainterp::JitCode = (**__dispatch_arc).clone();

            // Observer mode: the outer Rust mainloop runs the same opcode
            // body alongside this metainterp pass. The metainterp executes
            // each residual function-pointer call (BC_CALL_INT /
            // BC_RESIDUAL_CALL_VOID etc.) and pushes (func, args[, result])
            // onto OBSERVED_CALLS; the outer body, rewritten by `rewrite_body`
            // so each registered helper is wrapped in `consume_observed_*_call`,
            // replays the queued result instead of invoking the helper a
            // second time. The IR call op recorded above runs at compiled-
            // trace runtime; the outer/metainterp pair stays single-execution
            // per recording iter.
            //
            // Dispatch JitCode reads `program` and `pc` as caller-provided
            // IR arguments (matching the (Ref, Int) prefix of `reds` declared
            // by `__declare_jit_schema_*` and consumed by
            // `trace_jitcode_observer_with_args`).
            let mut __jitcode_args: ::std::vec::Vec<(
                majit_metainterp::JitArgKind,
                majit_ir::OpRef,
                i64,
            )> = ::std::vec::Vec::new();
            let __program_bits = program as *const #env_type as *const () as usize as i64;
            let __program_box = __ctx.const_ref(__program_bits);
            __jitcode_args.push((
                majit_metainterp::JitArgKind::Ref,
                __program_box,
                __program_bits,
            ));
            let __pc_bits = pc as i64;
            let __pc_box = __ctx.const_int(__pc_bits);
            __jitcode_args.push((
                majit_metainterp::JitArgKind::Int,
                __pc_box,
                __pc_bits,
            ));
            #push_virtualizable_argbox
            let __result = majit_metainterp::trace_jitcode_observer_with_args_and_runtime(
                __ctx,
                __sym,
                &__jitcode,
                pc,
                __runtime,
                &__jitcode_args,
            );
            if majit_metainterp::majit_log_enabled() && !matches!(__result, TraceAction::Continue) {
                eprintln!(
                    "[jit] trace action at pc={} -> {:?}",
                    pc,
                    __result
                );
            }
            __result
        }
    };

    quote! {
        /// Dispatch JitCode singleton builder.
        ///
        /// Builds the entire dispatch loop body (jit_merge_point + pre-dispatch
        /// ops + opcode fetch + dispatch chain + per-arm INLINE_CALL + loop
        /// close) as a single JitCode. `__trace_*` invokes this; the install
        /// pipeline registers it via `JitDriver::register_dispatch_jitcode`.
        ///
        /// Returns `Option<JitCode>`: `Some(jc)` when `lower_dispatch_body`
        /// succeeded at proc-macro time, `None` when the body shape was
        /// rejected (e.g. unrecognised inner control flow).  PyPy's
        /// `make_jitcodes()` / `pyjitpl.py:2255 finish_setup()` only
        /// install completed jitcodes — there is no "empty body installed
        /// as success" path.  The install pipeline at
        /// `codegen_state.rs:840` `if let Some(jc) = ... { register }`
        /// matches that lifecycle by skipping `register_dispatch_jitcode`
        /// when this returns `None`.
        #[allow(non_snake_case, unused_variables, unused_mut)]
        #[doc(hidden)]
        pub fn #dispatch_jitcode_fn_name(
            __asm: &mut majit_metainterp::Assembler,
            // jtransform.py:1704 portal_jd.index threaded as runtime param.
            __jdindex: i64,
        ) -> Option<majit_metainterp::JitCode> {
            if !#dispatch_lower_ok {
                // `lower_dispatch_body` rejected the body at proc-macro
                // time; surface as None so the install pipeline skips
                // `register_dispatch_jitcode` per PyPy parity.
                return None;
            }
            let mut __builder = majit_metainterp::JitCodeBuilder::new();
            let _live_offset_patch = __builder.live_placeholder();
            #dispatch_body
            __builder.finalize_liveness(__asm);
            Some(__builder.finish())
        }

        /// Pre-register the dispatch JitCode's per-marker liveness
        /// triples into the driver-shared `Assembler`, mirroring RPython
        /// `pyjitpl.py:2255 finish_setup`'s "all `-live-` entries land
        /// in `asm.all_liveness` before the snapshot" invariant.
        /// Invoked from `__JitMeta::install_canonical_liveness` exactly
        /// once at install time, before
        /// `JitDriver::install_canonical_liveness` snapshots
        /// `metainterp_sd.liveness_info`.
        #[allow(non_snake_case, unused_variables, unused_mut)]
        fn #prebuild_fn_name(__asm: &mut majit_metainterp::Assembler) {
            #dispatch_prebuild
        }

        /// Slice (audit Issue #5) — declare the dispatch JitCode's
        /// `(name, GreenType)` green schema and `(name, IR Type)` red
        /// schema on the JitDriver so `JitDriverStaticData::
        /// green_args_spec` reports STR/UNICODE subtypes and
        /// `green_kind_counts` / `red_kind_counts` reflect the real
        /// payload of `BC_JIT_MERGE_POINT`.  RPython
        /// `warmspot.py:663-665` derives the same `_green_args_spec`
        /// from the `JIT_ENTER_FUNCTYPE` signature; pyre derives it
        /// from the `lowerer.bindings` BindingKind plus
        /// `green_type_tags` (the `: str` / `: unicode` declarations)
        /// at `lower_dispatch_body` time.  No-op when the dispatch
        /// body failed to lower (the schema vectors are then empty).
        #[allow(non_snake_case, unused_variables, unused_mut)]
        fn #declare_schema_fn_name<S: majit_metainterp::JitState>(
            __driver: &mut majit_metainterp::JitDriver<S>,
        ) {
            let __greens: ::std::vec::Vec<(&str, majit_ir::GreenType)> = vec![
                #( (#dispatch_green_schema_names, #dispatch_green_schema_types) ),*
            ];
            let __reds: ::std::vec::Vec<(&str, majit_ir::Type)> = vec![
                #( (#dispatch_red_schema_names, #dispatch_red_schema_types) ),*
            ];
            __driver.declare_schema_typed(__greens, __reds);
        }

        #trace_fn_body
    }
}

pub(crate) fn find_dispatch_match(block: &syn::Block) -> Option<&syn::ExprMatch> {
    // Select the match with the most arms — the dispatch match has many
    // opcode arms while pre-dispatch branch matches have only a few.
    let mut all = Vec::new();
    collect_all_matches(block, &mut all);
    all.into_iter().max_by_key(|m| m.arms.len())
}

fn collect_all_matches<'a>(block: &'a syn::Block, out: &mut Vec<&'a syn::ExprMatch>) {
    for stmt in &block.stmts {
        collect_matches_in_stmt(stmt, out);
    }
}

fn collect_matches_in_stmt<'a>(stmt: &'a syn::Stmt, out: &mut Vec<&'a syn::ExprMatch>) {
    match stmt {
        syn::Stmt::Expr(expr, _) => collect_matches_in_expr(expr, out),
        syn::Stmt::Local(local) => {
            if let Some(init) = &local.init {
                collect_matches_in_expr(&init.expr, out);
            }
        }
        _ => {}
    }
}

fn collect_matches_in_expr<'a>(expr: &'a syn::Expr, out: &mut Vec<&'a syn::ExprMatch>) {
    match expr {
        syn::Expr::Match(m) => out.push(m),
        syn::Expr::While(w) => collect_all_matches(&w.body, out),
        syn::Expr::Loop(l) => collect_all_matches(&l.body, out),
        syn::Expr::Block(b) => collect_all_matches(&b.block, out),
        syn::Expr::If(i) => {
            collect_all_matches(&i.then_branch, out);
            if let Some((_, else_expr)) = &i.else_branch {
                collect_matches_in_expr(else_expr, out);
            }
        }
        _ => {}
    }
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

/// `true` iff `stmt`'s expression subtree contains `target_match`.
pub(crate) fn stmt_contains_match(stmt: &Stmt, target_match: &ExprMatch) -> bool {
    match stmt {
        Stmt::Expr(expr, _) => expr_contains_match(expr, target_match),
        Stmt::Local(local) => local
            .init
            .as_ref()
            .map(|init| expr_contains_match(&init.expr, target_match))
            .unwrap_or(false),
        _ => false,
    }
}

pub(crate) fn block_contains_match(block: &Block, target_match: &ExprMatch) -> bool {
    block
        .stmts
        .iter()
        .any(|s| stmt_contains_match(s, target_match))
}

fn expr_contains_match(expr: &Expr, target_match: &ExprMatch) -> bool {
    match expr {
        Expr::Match(m) => std::ptr::eq(m, target_match),
        Expr::While(w) => block_contains_match(&w.body, target_match),
        Expr::Loop(l) => block_contains_match(&l.body, target_match),
        Expr::Block(b) => block_contains_match(&b.block, target_match),
        Expr::If(i) => {
            block_contains_match(&i.then_branch, target_match)
                || i.else_branch
                    .as_ref()
                    .map(|(_, e)| expr_contains_match(e, target_match))
                    .unwrap_or(false)
        }
        _ => false,
    }
}

/// Check if a call expression's function path is a promote call.
///
/// Matches: `promote`, `hint_promote`, `jit::promote`,
/// `majit_metainterp::jit::promote`.
pub(crate) fn is_promote_call_path(func: &syn::Expr) -> bool {
    let syn::Expr::Path(func_path) = func else {
        return false;
    };
    let segments: Vec<_> = func_path
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    match segments.as_slice() {
        [name] => name == "promote" || name == "hint_promote",
        [ns, name] => name == "promote" && ns == "jit",
        [_, ns, name] => name == "promote" && ns == "jit",
        _ => false,
    }
}

/// Check if a call expression's function path is a `jit::assert_not_none`
/// call.
///
/// Matches: `assert_not_none`, `jit::assert_not_none`,
/// `majit_metainterp::jit::assert_not_none`.  Mirrors RPython
/// `rtyper/debug.py:23 ll_assert_not_none` recognition.
pub(crate) fn is_assert_not_none_call_path(func: &syn::Expr) -> bool {
    let syn::Expr::Path(func_path) = func else {
        return false;
    };
    let segments: Vec<_> = func_path
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    match segments.as_slice() {
        [name] => name == "assert_not_none",
        [ns, name] => name == "assert_not_none" && ns == "jit",
        [_, ns, name] => name == "assert_not_none" && ns == "jit",
        _ => false,
    }
}

/// Check if a call expression's function path is a `jit::record_exact_class`
/// call.
///
/// Matches: `record_exact_class`, `jit::record_exact_class`,
/// `majit_metainterp::jit::record_exact_class`.  Mirrors RPython
/// `rlib/jit.py:1181 jit.record_exact_class` recognition.
pub(crate) fn is_record_exact_class_call_path(func: &syn::Expr) -> bool {
    let syn::Expr::Path(func_path) = func else {
        return false;
    };
    let segments: Vec<_> = func_path
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    match segments.as_slice() {
        [name] => name == "record_exact_class",
        [ns, name] => name == "record_exact_class" && ns == "jit",
        [_, ns, name] => name == "record_exact_class" && ns == "jit",
        _ => false,
    }
}
