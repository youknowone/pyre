//! AST transformation utilities for JIT mainloop generation.
//!
//! Rewrites interpreter source patterns (storage/selected references,
//! branch arms) into JIT-compatible forms.

use proc_macro2::TokenStream;
use quote::quote;

/// Rewrite `storage.xxx` to `state.storage.xxx` and `selected` to `state.selected`
/// in a token stream.
///
/// Only replaces bare identifiers, not already-prefixed ones. The patterns
/// handled are the common interpreter state field access patterns:
/// - `storage.get_mut(selected)` -> `state.storage.get_mut(state.selected)`
/// - `storage.get(selected)` -> `state.storage.get(state.selected)`
/// - `storage.get_mut(target)` -> `state.storage.get_mut(target)`
/// - `selected = value` -> `state.selected = value`
/// - `selected == target` -> `state.selected == target`
pub fn rewrite_storage_refs(tokens: TokenStream) -> TokenStream {
    let s = tokens.to_string();
    let s = s
        .replace(
            "storage . get_mut (selected)",
            "state . storage . get_mut (state . selected)",
        )
        .replace(
            "storage . get (selected)",
            "state . storage . get (state . selected)",
        )
        .replace(
            "storage . get_mut (target)",
            "state . storage . get_mut (target)",
        )
        .replace("selected = value", "state . selected = value")
        .replace("selected == target", "state . selected == target");

    s.parse()
        .unwrap_or_else(|e| panic!("rewrite parse error: {e}\n{s}"))
}

/// Transform a branch arm for JIT-enabled execution.
///
/// Given the pattern and body of a branch/jump arm (containing `get_label` +
/// `continue`), generates a JIT version that calls `run_jit_back_edge` at
/// backward jump sites.
pub fn transform_branch_arm(pat: &syn::Pat, _body: &syn::Expr) -> TokenStream {
    // The original branch arm body contains:
    //   let jump = match op { ... };
    //   if jump { pc = program.get_label(pc); continue; }
    //
    // We rewrite the `if jump` block to:
    //   if jump {
    //       let target = program.get_label(pc);
    //       if target <= pc {
    //           if let Some(jit_pc) = run_jit_back_edge(...) { pc = jit_pc; ...; continue; }
    //       }
    //       pc = target; continue;
    //   }
    //
    // The inner match (for BRZ pop) needs storage->state.storage rewriting.
    quote! {
        #pat => {
            let jump = match op {
                OP_BRPOP1 | OP_BRPOP2 => !stackok,
                OP_JMP => true,
                OP_BRZ => {
                    let top = state.storage.get_mut(state.selected).pop();
                    val_is_zero(&top)
                }
                _ => unreachable!(),
            };
            if jump {
                let target = program.get_label(pc);
                if target <= pc {
                    if let Some(jit_pc) = run_jit_back_edge(
                        &mut driver, target, &mut state, program,
                        || { aheui_io::output_flush(); }
                    ) {
                        pc = jit_pc;
                        stacksize = state.storage.get(state.selected).len() as i32;
                        continue;
                    }
                }
                pc = target;
                continue;
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewrite_storage_get_mut_selected() {
        let input: TokenStream = "storage . get_mut (selected) . push(v)".parse().unwrap();
        let output = rewrite_storage_refs(input);
        let s = output.to_string();
        assert!(
            s.contains("state . storage . get_mut (state . selected)"),
            "got: {s}"
        );
    }

    #[test]
    fn test_rewrite_storage_get_selected() {
        let input: TokenStream = "storage . get (selected) . len()".parse().unwrap();
        let output = rewrite_storage_refs(input);
        let s = output.to_string();
        assert!(
            s.contains("state . storage . get (state . selected)"),
            "got: {s}"
        );
    }

    #[test]
    fn test_rewrite_storage_get_mut_target() {
        let input: TokenStream = "storage . get_mut (target) . push(v)".parse().unwrap();
        let output = rewrite_storage_refs(input);
        let s = output.to_string();
        assert!(
            s.contains("state . storage . get_mut (target)"),
            "got: {s}"
        );
        // Should NOT add state.target
        assert!(!s.contains("state . target"), "got: {s}");
    }

    #[test]
    fn test_rewrite_selected_assign() {
        let input: TokenStream = "selected = value".parse().unwrap();
        let output = rewrite_storage_refs(input);
        let s = output.to_string();
        assert!(s.contains("state . selected = value"), "got: {s}");
    }

    #[test]
    fn test_rewrite_selected_compare() {
        let input: TokenStream = "selected == target".parse().unwrap();
        let output = rewrite_storage_refs(input);
        let s = output.to_string();
        assert!(s.contains("state . selected == target"), "got: {s}");
    }

    #[test]
    fn test_transform_branch_arm_compiles() {
        // Just verify it produces valid tokens
        let arm: syn::Arm =
            syn::parse_str("OP_BRPOP1 | OP_BRPOP2 | OP_JMP | OP_BRZ => {}").unwrap();
        let body: syn::Expr = syn::parse_str("{}").unwrap();
        let tokens = transform_branch_arm(&arm.pat, &body);
        let s = tokens.to_string();
        assert!(s.contains("run_jit_back_edge"), "got: {s}");
        assert!(s.contains("state . storage"), "got: {s}");
    }
}
