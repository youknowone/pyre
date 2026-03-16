//! Source parsing: extract opcode dispatch, trait impls, type layouts.

use syn::{File, Item, ItemFn, ItemImpl, ItemStruct, visit::Visit};

use crate::{FieldInfo, MethodInfo, OpcodeArm, TraitImplInfo, TypeLayout};

/// Parsed representation of an interpreter source file.
pub struct ParsedInterpreter {
    pub file: File,
}

/// Parse a bundled Rust source file.
pub fn parse_source(source: &str) -> ParsedInterpreter {
    let file = syn::parse_file(source).expect("failed to parse bundled source");
    ParsedInterpreter { file }
}

/// Extract struct layouts from the parsed source.
pub fn extract_type_layouts(parsed: &ParsedInterpreter) -> Vec<TypeLayout> {
    let mut layouts = Vec::new();
    for item in &parsed.file.items {
        if let Item::Struct(s) = item {
            let name = s.ident.to_string();
            let fields = match &s.fields {
                syn::Fields::Named(named) => named
                    .named
                    .iter()
                    .map(|f| {
                        let ty = &f.ty;
                        FieldInfo {
                            name: f.ident.as_ref().map_or("_".into(), |i| i.to_string()),
                            ty: quote::quote!(#ty).to_string(),
                            offset_expr: None,
                        }
                    })
                    .collect(),
                _ => Vec::new(),
            };
            if !fields.is_empty() {
                layouts.push(TypeLayout { name, fields });
            }
        }
    }
    layouts
}

/// Extract trait implementations from the parsed source.
pub fn extract_trait_impls(parsed: &ParsedInterpreter) -> Vec<TraitImplInfo> {
    let mut impls = Vec::new();
    for item in &parsed.file.items {
        if let Item::Impl(impl_block) = item {
            if let Some((_, trait_path, _)) = &impl_block.trait_ {
                let trait_name = quote::quote!(#trait_path).to_string();
                let self_ty = &impl_block.self_ty;
                let for_type = quote::quote!(#self_ty).to_string();
                let methods: Vec<MethodInfo> = impl_block
                    .items
                    .iter()
                    .filter_map(|item| {
                        if let syn::ImplItem::Fn(method) = item {
                            Some(MethodInfo {
                                name: method.sig.ident.to_string(),
                                body_summary: summarize_block(&method.block),
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                if !methods.is_empty() {
                    impls.push(TraitImplInfo {
                        trait_name,
                        for_type,
                        methods,
                    });
                }
            }
        }
    }
    impls
}

/// Extract opcode dispatch match arms from execute_opcode_step.
pub fn extract_opcode_dispatch(
    parsed: &ParsedInterpreter,
    trait_impls: &[TraitImplInfo],
) -> Vec<OpcodeArm> {
    let mut arms = Vec::new();

    // Find execute_opcode_step function
    for item in &parsed.file.items {
        if let Item::Fn(func) = item {
            if func.sig.ident == "execute_opcode_step" {
                arms = extract_match_arms(func);
                break;
            }
        }
    }

    // Resolve trait method calls to concrete implementations
    for arm in &mut arms {
        for call in &arm.handler_calls {
            // Try to find the concrete implementation in trait_impls
            for impl_info in trait_impls {
                for method in &impl_info.methods {
                    if method.name == *call {
                        arm.trace_pattern =
                            crate::patterns::classify_method_body(&method.body_summary);
                        break;
                    }
                }
            }
        }
    }

    arms
}

/// Extract match arms from a function containing a match on instruction.
fn extract_match_arms(func: &ItemFn) -> Vec<OpcodeArm> {
    let mut collector = MatchArmCollector { arms: Vec::new() };
    collector.visit_item_fn(func);
    collector.arms
}

struct MatchArmCollector {
    arms: Vec<OpcodeArm>,
}

impl<'ast> Visit<'ast> for MatchArmCollector {
    fn visit_expr_match(&mut self, expr: &'ast syn::ExprMatch) {
        for arm in &expr.arms {
            let pat = &arm.pat;
            let pattern = quote::quote!(#pat).to_string();
            let handler_calls = extract_method_calls(&arm.body);
            self.arms.push(OpcodeArm {
                pattern,
                handler_calls,
                trace_pattern: None,
            });
        }
    }
}

/// Extract method call names from an expression.
fn extract_method_calls(expr: &syn::Expr) -> Vec<String> {
    let mut calls = Vec::new();
    let mut collector = CallCollector { calls: &mut calls };
    syn::visit::visit_expr(&mut collector, expr);
    calls
}

struct CallCollector<'a> {
    calls: &'a mut Vec<String>,
}

impl<'ast, 'a> Visit<'ast> for CallCollector<'a> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.calls.push(call.method.to_string());
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*call.func {
            if let Some(segment) = path.path.segments.last() {
                self.calls.push(segment.ident.to_string());
            }
        }
        syn::visit::visit_expr_call(self, call);
    }
}

/// Create a brief summary of a block's content.
fn summarize_block(block: &syn::Block) -> String {
    let tokens = quote::quote!(#block);
    let s = tokens.to_string();
    if s.len() > 200 {
        format!("{}...", &s[..200])
    } else {
        s
    }
}
