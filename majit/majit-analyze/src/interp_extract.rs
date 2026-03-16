//! Extract interpreter structure from Rust source code.
//!
//! Generic utilities for finding mainloop functions, opcode dispatch match
//! expressions, and analyzing value types for overflow/bigint patterns.

use quote::quote;
use syn::visit::Visit;
use syn::{Expr, ExprMatch, ExprMethodCall, File, ItemFn};

/// A mapping from an interpreter's binary operation method name to its JIT opcode.
#[derive(Debug, Clone)]
pub struct BinopMapping {
    pub method: String,
    pub opcode: String,
}

/// Find a function named `mainloop` in a parsed AST.
pub fn find_mainloop(file: &File) -> Option<&ItemFn> {
    file.items.iter().find_map(|item| {
        if let syn::Item::Fn(f) = item {
            if f.sig.ident == "mainloop" {
                return Some(f);
            }
        }
        None
    })
}

/// Find the opcode dispatch `match` expression within a function.
///
/// Looks for a match whose first arm pattern contains `OP_` — this is the
/// standard convention for interpreter opcode dispatch.
pub fn find_opcode_match(func: &ItemFn) -> Option<&ExprMatch> {
    struct Finder<'a> {
        result: Option<&'a ExprMatch>,
    }
    impl<'ast> Visit<'ast> for Finder<'ast> {
        fn visit_expr_match(&mut self, node: &'ast ExprMatch) {
            // The opcode match is `match op { OP_ADD => ... }`
            if self.result.is_none() {
                let first_arm = node.arms.first().map(|a| quote!(#a.pat).to_string());
                if first_arm.as_deref().is_some_and(|s| s.contains("OP_")) {
                    self.result = Some(node);
                    return;
                }
            }
            syn::visit::visit_expr_match(self, node);
        }
    }
    let mut finder = Finder { result: None };
    finder.visit_item_fn(func);
    finder.result
}

/// Analyze a value source file for checked/overflow arithmetic patterns.
///
/// Returns a fixed set of binop mappings. The analysis detects whether the
/// value type uses checked arithmetic (bigint) or direct i32 operations.
pub fn analyze_value_overflow(source: &str) -> Vec<BinopMapping> {
    let file: File = syn::parse_str(source).expect("failed to parse value source");
    let has_bigint =
        function_uses_checked_ops(&file, "val_add") || function_uses_checked_ops(&file, "val_mul");
    eprintln!("[majit-analyze] bigint detected: {has_bigint} -- using i32 acceleration");
    vec![
        BinopMapping {
            method: "add".into(),
            opcode: "IntAdd".into(),
        },
        BinopMapping {
            method: "sub".into(),
            opcode: "IntSub".into(),
        },
        BinopMapping {
            method: "mul".into(),
            opcode: "IntMul".into(),
        },
        BinopMapping {
            method: "div".into(),
            opcode: "IntFloorDiv".into(),
        },
        BinopMapping {
            method: "modulo".into(),
            opcode: "IntMod".into(),
        },
        BinopMapping {
            method: "cmp".into(),
            opcode: "IntGe".into(),
        },
    ]
}

/// Check if a function uses `checked_*` method calls (indicating bigint overflow handling).
pub fn function_uses_checked_ops(file: &File, fn_name: &str) -> bool {
    // Direct check: look for checked_* method calls inside the target function
    struct V {
        target: String,
        inside: bool,
        found: bool,
    }
    impl<'ast> Visit<'ast> for V {
        fn visit_item_fn(&mut self, n: &'ast ItemFn) {
            if n.sig.ident == self.target {
                self.inside = true;
                syn::visit::visit_item_fn(self, n);
                self.inside = false;
            } else {
                syn::visit::visit_item_fn(self, n);
            }
        }
        fn visit_expr_method_call(&mut self, n: &'ast ExprMethodCall) {
            if self.inside && n.method.to_string().starts_with("checked_") {
                self.found = true;
            }
            syn::visit::visit_expr_method_call(self, n);
        }
    }
    let mut v = V {
        target: fn_name.into(),
        inside: false,
        found: false,
    };
    v.visit_file(file);
    if v.found {
        return true;
    }
    // Indirect check: does the function call binop_fast?
    struct V2 {
        target: String,
        inside: bool,
        found: bool,
    }
    impl<'ast> Visit<'ast> for V2 {
        fn visit_item_fn(&mut self, n: &'ast ItemFn) {
            if n.sig.ident == self.target {
                self.inside = true;
                syn::visit::visit_item_fn(self, n);
                self.inside = false;
            } else {
                syn::visit::visit_item_fn(self, n);
            }
        }
        fn visit_expr(&mut self, n: &'ast Expr) {
            if self.inside {
                if let Expr::Call(c) = n {
                    if quote!(#c).to_string().contains("binop_fast") {
                        self.found = true;
                    }
                }
            }
            syn::visit::visit_expr(self, n);
        }
    }
    let mut v2 = V2 {
        target: fn_name.into(),
        inside: false,
        found: false,
    };
    v2.visit_file(file);
    v2.found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_mainloop_present() {
        let src = r#"
            fn helper() {}
            fn mainloop() {
                loop {}
            }
        "#;
        let file: File = syn::parse_str(src).unwrap();
        let ml = find_mainloop(&file);
        assert!(ml.is_some());
        assert_eq!(ml.unwrap().sig.ident, "mainloop");
    }

    #[test]
    fn test_find_mainloop_absent() {
        let src = r#"
            fn not_mainloop() {}
        "#;
        let file: File = syn::parse_str(src).unwrap();
        assert!(find_mainloop(&file).is_none());
    }

    #[test]
    fn test_find_opcode_match() {
        let src = r#"
            fn mainloop() {
                let op = 0;
                match op {
                    OP_ADD => {},
                    OP_SUB => {},
                    _ => {},
                }
            }
        "#;
        let file: File = syn::parse_str(src).unwrap();
        let ml = find_mainloop(&file).unwrap();
        let m = find_opcode_match(ml);
        assert!(m.is_some());
        assert_eq!(m.unwrap().arms.len(), 3);
    }

    #[test]
    fn test_find_opcode_match_nested() {
        let src = r#"
            fn mainloop() {
                loop {
                    let op = get_op();
                    match op {
                        OP_NOP => {},
                        _ => {},
                    }
                }
            }
        "#;
        let file: File = syn::parse_str(src).unwrap();
        let ml = find_mainloop(&file).unwrap();
        assert!(find_opcode_match(ml).is_some());
    }

    #[test]
    fn test_function_uses_checked_ops_direct() {
        let src = r#"
            fn val_add(a: i32, b: i32) -> i32 {
                a.checked_add(b).unwrap()
            }
        "#;
        let file: File = syn::parse_str(src).unwrap();
        assert!(function_uses_checked_ops(&file, "val_add"));
        assert!(!function_uses_checked_ops(&file, "val_sub"));
    }

    #[test]
    fn test_function_uses_checked_ops_indirect() {
        let src = r#"
            fn val_add(a: i32, b: i32) -> i32 {
                binop_fast(a, b)
            }
        "#;
        let file: File = syn::parse_str(src).unwrap();
        assert!(function_uses_checked_ops(&file, "val_add"));
    }

    #[test]
    fn test_analyze_value_overflow() {
        let src = r#"
            fn val_add(a: i32, b: i32) -> i32 { a + b }
            fn val_mul(a: i32, b: i32) -> i32 { a * b }
        "#;
        let binops = analyze_value_overflow(src);
        assert_eq!(binops.len(), 6);
        assert_eq!(binops[0].method, "add");
        assert_eq!(binops[0].opcode, "IntAdd");
    }
}
