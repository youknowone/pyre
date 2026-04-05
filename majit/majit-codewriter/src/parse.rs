//! Source parsing: extract opcode dispatch and trait impls.

use crate::{MethodInfo, TraitImplInfo};
use serde::{Deserialize, Serialize};
use syn::{ExprMatch, File, Item, ItemFn, Pat, Path, visit::Visit};

/// Raw opcode-dispatch arm extracted from the interpreter match.
///
/// This is the canonical parse/front-end view of opcode dispatch before
/// graph/pipeline classification is attached.
#[derive(Debug, Clone)]
pub struct ExtractedOpcodeArm {
    pub selector: OpcodeDispatchSelector,
    pub handler_calls: Vec<ExtractedHandlerCall>,
    /// Semantic graph of the match arm body.
    /// This is the handler's own graph — the primary input for
    /// jtransform/flatten. handler_calls are metadata only.
    pub body_graph: Option<crate::model::FunctionGraph>,
}

#[derive(Debug, Clone, Default)]
pub struct ReceiverTraitBindings {
    pub traits_by_receiver: std::collections::HashMap<String, Vec<String>>,
    pub type_root_by_receiver: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CallPath {
    pub segments: Vec<String>,
}

impl CallPath {
    pub fn from_segments<I, S>(segments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            segments: segments.into_iter().map(Into::into).collect(),
        }
    }

    pub fn last_segment(&self) -> Option<&str> {
        self.segments.last().map(String::as_str)
    }

    pub fn canonical_key(&self) -> String {
        self.segments.join("::")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpcodeDispatchSelector {
    Path(CallPath),
    Wildcard,
    Or(Vec<OpcodeDispatchSelector>),
    Unsupported,
}

impl OpcodeDispatchSelector {
    pub fn canonical_key(&self) -> String {
        match self {
            Self::Path(path) => path.canonical_key(),
            Self::Wildcard => "_".into(),
            Self::Or(cases) => cases
                .iter()
                .map(Self::canonical_key)
                .collect::<Vec<_>>()
                .join(" | "),
            Self::Unsupported => "<unsupported>".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractedHandlerCall {
    Method {
        name: String,
        receiver_root: Option<String>,
    },
    FunctionPath(CallPath),
    UnsupportedFunctionExpr,
}

#[derive(Debug, Clone)]
pub struct InherentMethodInfo {
    pub for_type: String,
    pub self_ty_root: Option<String>,
    pub name: String,
    pub graph: crate::model::FunctionGraph,
    /// RPython: op.result.concretetype — return type for array identity.
    pub return_type: Option<String>,
    /// RPython: function-level JIT hints (elidable, close_stack, etc.).
    pub hints: Vec<String>,
}

/// Parsed representation of an interpreter source file.
pub struct ParsedInterpreter {
    pub file: File,
}

/// Parse a bundled Rust source file.
pub fn parse_source(source: &str) -> ParsedInterpreter {
    let file = syn::parse_file(source).expect("failed to parse bundled source");
    ParsedInterpreter { file }
}

/// Find a top-level function by exact name in the parsed source.
pub(crate) fn find_function<'a>(parsed: &'a ParsedInterpreter, name: &str) -> Option<&'a ItemFn> {
    find_function_in_file(&parsed.file, name)
}

/// Find a top-level function by exact name in a parsed file.
pub(crate) fn find_function_in_file<'a>(file: &'a File, name: &str) -> Option<&'a ItemFn> {
    file.items.iter().find_map(|item| {
        if let Item::Fn(func) = item {
            (func.sig.ident == name).then_some(func)
        } else {
            None
        }
    })
}

/// Find an opcode-dispatch `match` expression within a function.
fn find_opcode_match(func: &ItemFn) -> Option<&ExprMatch> {
    struct Finder<'a> {
        result: Option<&'a ExprMatch>,
    }

    impl<'ast> Visit<'ast> for Finder<'ast> {
        fn visit_expr_match(&mut self, node: &'ast ExprMatch) {
            if self.result.is_none() && node.arms.first().is_some_and(is_opcode_pattern) {
                self.result = Some(node);
                return;
            }
            syn::visit::visit_expr_match(self, node);
        }
    }

    let mut finder = Finder { result: None };
    finder.visit_item_fn(func);
    finder.result
}

/// Find the canonical opcode-dispatch match in a parsed interpreter source.
///
/// This is the public parse/front-end helper for consumers that still need the
/// raw `match` AST rather than the extracted `ExtractedOpcodeArm` view.
pub fn find_opcode_dispatch_match(parsed: &ParsedInterpreter) -> Option<&ExprMatch> {
    find_function(parsed, "mainloop").and_then(find_opcode_match)
}

/// Extract trait implementations AND trait default methods from the parsed source.
/// Recurses into `Item::Mod` for whole-program visibility (RPython parity).
pub fn extract_trait_impls(
    parsed: &ParsedInterpreter,
    struct_fields: &crate::front::StructFieldRegistry,
    fn_return_types: &std::collections::HashMap<String, String>,
) -> Vec<TraitImplInfo> {
    let mut impls = Vec::new();
    collect_trait_impls_from_items(
        &parsed.file.items,
        "",
        struct_fields,
        fn_return_types,
        &mut impls,
    );
    impls
}

fn collect_trait_impls_from_items(
    items: &[Item],
    prefix: &str,
    struct_fields: &crate::front::StructFieldRegistry,
    fn_return_types: &std::collections::HashMap<String, String>,
    impls: &mut Vec<TraitImplInfo>,
) {
    for item in items {
        match item {
            // Concrete trait impls (impl Trait for Type)
            Item::Impl(impl_block) => {
                if let Some((_, trait_path, _)) = &impl_block.trait_ {
                    let trait_name = canonical_path_name(trait_path);
                    let self_ty = &impl_block.self_ty;
                    let for_type = canonical_type_name(self_ty);
                    // Qualify bare type name with module prefix (RPython: unique type identity).
                    let self_ty_root = type_root_ident(self_ty)
                        .map(|t| crate::front::ast::qualify_type_name(&t, prefix));
                    let methods: Vec<MethodInfo> = impl_block
                        .items
                        .iter()
                        .filter_map(|item| {
                            if let syn::ImplItem::Fn(method) = item {
                                let (graph, hints) = {
                                    let fake_fn = syn::ItemFn {
                                        attrs: method.attrs.clone(),
                                        vis: syn::Visibility::Inherited,
                                        sig: method.sig.clone(),
                                        block: Box::new(method.block.clone()),
                                    };
                                    let sf =
                                        crate::front::ast::build_function_graph_with_self_ty_pub(
                                            &fake_fn,
                                            self_ty_root.clone(),
                                            struct_fields,
                                            fn_return_types,
                                            prefix,
                                        );
                                    (Some(sf.graph), sf.hints)
                                };
                                let return_type = match &method.sig.output {
                                    syn::ReturnType::Type(_, ty) => {
                                        crate::front::ast::full_type_string(ty)
                                    }
                                    syn::ReturnType::Default => None,
                                };
                                Some(MethodInfo {
                                    name: method.sig.ident.to_string(),
                                    graph,
                                    return_type,
                                    hints,
                                })
                            } else {
                                None
                            }
                        })
                        .collect();
                    impls.push(TraitImplInfo {
                        trait_name,
                        for_type,
                        self_ty_root,
                        methods,
                    });
                }
            }
            // Trait definitions with default methods
            Item::Trait(trait_def) => {
                let trait_name = trait_def.ident.to_string();
                let methods: Vec<MethodInfo> = trait_def
                    .items
                    .iter()
                    .filter_map(|item| {
                        if let syn::TraitItem::Fn(method) = item {
                            method.default.as_ref().map(|block| {
                                let fake_fn = syn::ItemFn {
                                    attrs: method.attrs.clone(),
                                    vis: syn::Visibility::Inherited,
                                    sig: method.sig.clone(),
                                    block: Box::new(block.clone()),
                                };
                                let sf = crate::front::ast::build_function_graph_with_self_ty_pub(
                                    &fake_fn,
                                    None,
                                    struct_fields,
                                    fn_return_types,
                                    prefix,
                                );
                                let return_type = match &method.sig.output {
                                    syn::ReturnType::Type(_, ty) => {
                                        crate::front::ast::full_type_string(ty)
                                    }
                                    syn::ReturnType::Default => None,
                                };
                                MethodInfo {
                                    name: method.sig.ident.to_string(),
                                    graph: Some(sf.graph),
                                    return_type,
                                    hints: sf.hints,
                                }
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                if !methods.is_empty() {
                    impls.push(TraitImplInfo {
                        trait_name: trait_name.clone(),
                        for_type: format!("<default methods of {}>", trait_name),
                        self_ty_root: None,
                        methods,
                    });
                }
            }
            // Recurse into module blocks with qualified prefix.
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_trait_impls_from_items(
                        sub_items,
                        &mod_prefix,
                        struct_fields,
                        fn_return_types,
                        impls,
                    );
                }
            }
            _ => {}
        }
    }
}

/// Extract inherent impl methods (impl Type { ... }) as canonical call targets.
/// Recurses into `Item::Mod` for whole-program visibility (RPython parity).
pub fn extract_inherent_impl_methods(
    parsed: &ParsedInterpreter,
    struct_fields: &crate::front::StructFieldRegistry,
    fn_return_types: &std::collections::HashMap<String, String>,
) -> Vec<InherentMethodInfo> {
    let mut methods = Vec::new();
    collect_inherent_methods_from_items(
        &parsed.file.items,
        "",
        struct_fields,
        fn_return_types,
        &mut methods,
    );
    methods
}

fn collect_inherent_methods_from_items(
    items: &[Item],
    prefix: &str,
    struct_fields: &crate::front::StructFieldRegistry,
    fn_return_types: &std::collections::HashMap<String, String>,
    methods: &mut Vec<InherentMethodInfo>,
) {
    for item in items {
        match item {
            Item::Impl(impl_block) => {
                if impl_block.trait_.is_some() {
                    continue;
                }
                let for_type = canonical_type_name(&impl_block.self_ty);
                // Qualify bare type name with module prefix (RPython: unique type identity).
                let self_ty_root = type_root_ident(&impl_block.self_ty)
                    .map(|t| crate::front::ast::qualify_type_name(&t, prefix));
                for sub in &impl_block.items {
                    if let syn::ImplItem::Fn(method) = sub {
                        let fake_fn = syn::ItemFn {
                            attrs: method.attrs.clone(),
                            vis: syn::Visibility::Inherited,
                            sig: method.sig.clone(),
                            block: Box::new(method.block.clone()),
                        };
                        let sf = crate::front::ast::build_function_graph_with_self_ty_pub(
                            &fake_fn,
                            self_ty_root.clone(),
                            struct_fields,
                            fn_return_types,
                            prefix,
                        );
                        let return_type = match &method.sig.output {
                            syn::ReturnType::Type(_, ty) => crate::front::ast::full_type_string(ty),
                            syn::ReturnType::Default => None,
                        };
                        methods.push(InherentMethodInfo {
                            for_type: for_type.clone(),
                            self_ty_root: self_ty_root.clone(),
                            name: method.sig.ident.to_string(),
                            graph: sf.graph,
                            return_type,
                            hints: sf.hints,
                        });
                    }
                }
            }
            // Recurse into module blocks with qualified prefix.
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_inherent_methods_from_items(
                        sub_items,
                        &mod_prefix,
                        struct_fields,
                        fn_return_types,
                        methods,
                    );
                }
            }
            _ => {}
        }
    }
}

/// Extract canonical opcode dispatch arms from `execute_opcode_step`.
///
/// This preserves source-level match structure and handler calls so canonical
/// graph/pipeline consumers can resolve and classify these arms directly.
pub fn extract_opcode_dispatch_arms(parsed: &ParsedInterpreter) -> Vec<ExtractedOpcodeArm> {
    for item in &parsed.file.items {
        if let Item::Fn(func) = item {
            if func.sig.ident == "execute_opcode_step" {
                return extract_match_arms(func);
            }
        }
    }

    Vec::new()
}

/// Extract receiver -> trait bounds for `execute_opcode_step`.
///
/// This lets canonical dispatch resolution follow generic receiver methods
/// through the trait that actually defines their default bodies.
pub fn extract_opcode_dispatch_receiver_traits(
    parsed: &ParsedInterpreter,
) -> ReceiverTraitBindings {
    for item in &parsed.file.items {
        if let Item::Fn(func) = item {
            if func.sig.ident == "execute_opcode_step" {
                return extract_receiver_trait_bindings(func);
            }
        }
    }
    ReceiverTraitBindings::default()
}

/// Collect canonical function names and graphs for the active pipeline path.
pub fn collect_function_graphs(
    parsed: &ParsedInterpreter,
    graphs: &mut std::collections::HashMap<CallPath, crate::model::FunctionGraph>,
) {
    for item in &parsed.file.items {
        if let Item::Fn(func) = item {
            let name = func.sig.ident.to_string();
            let sf = crate::front::ast::build_function_graph_pub(func);
            graphs.insert(CallPath::from_segments([name.clone()]), sf.graph.clone());
            graphs.insert(CallPath::from_segments(["crate", name.as_str()]), sf.graph);
        }
    }
}

/// Extract match arms from a function containing a match on instruction.
fn extract_match_arms(func: &ItemFn) -> Vec<ExtractedOpcodeArm> {
    let mut collector = MatchArmCollector { arms: Vec::new() };
    collector.visit_item_fn(func);
    collector.arms
}

fn is_opcode_pattern(arm: &syn::Arm) -> bool {
    pattern_is_opcode_dispatch(&arm.pat)
}

fn pattern_is_opcode_dispatch(pat: &Pat) -> bool {
    match pat {
        Pat::Ident(pat) => pat.ident.to_string().starts_with("OP_"),
        Pat::Path(path) => path_is_opcode_dispatch(&path.path),
        Pat::Struct(pat) => path_is_opcode_dispatch(&pat.path),
        Pat::TupleStruct(pat) => path_is_opcode_dispatch(&pat.path),
        Pat::Tuple(pat) => pat.elems.iter().any(pattern_is_opcode_dispatch),
        Pat::Or(pat) => pat.cases.iter().any(pattern_is_opcode_dispatch),
        _ => false,
    }
}

fn path_is_opcode_dispatch(path: &Path) -> bool {
    let last = path
        .segments
        .last()
        .map(|segment| segment.ident.to_string());
    if let Some(last) = last {
        if last.starts_with("OP_") {
            return true;
        }
    }
    path.segments
        .iter()
        .any(|segment| segment.ident == "Instruction")
}

struct MatchArmCollector {
    arms: Vec<ExtractedOpcodeArm>,
}

impl<'ast> Visit<'ast> for MatchArmCollector {
    fn visit_expr_match(&mut self, expr: &'ast syn::ExprMatch) {
        for arm in &expr.arms {
            let handler_calls = extract_handler_calls(&arm.body);
            let selector = extract_opcode_dispatch_selector(&arm.pat);
            // Build semantic graph from the arm body expression.
            let body_graph = {
                let name = selector.canonical_key();
                let mut graph = crate::model::FunctionGraph::new(name);
                crate::front::ast::lower_expr_into_graph(&mut graph, &arm.body);
                Some(graph)
            };
            self.arms.push(ExtractedOpcodeArm {
                selector,
                handler_calls,
                body_graph,
            });
        }
    }
}

fn extract_opcode_dispatch_selector(pat: &Pat) -> OpcodeDispatchSelector {
    match pat {
        Pat::Ident(pat) => {
            OpcodeDispatchSelector::Path(CallPath::from_segments([pat.ident.to_string()]))
        }
        Pat::Path(path) => OpcodeDispatchSelector::Path(CallPath::from_segments(
            path.path.segments.iter().map(|seg| seg.ident.to_string()),
        )),
        Pat::Struct(pat) => OpcodeDispatchSelector::Path(CallPath::from_segments(
            pat.path.segments.iter().map(|seg| seg.ident.to_string()),
        )),
        Pat::TupleStruct(pat) => OpcodeDispatchSelector::Path(CallPath::from_segments(
            pat.path.segments.iter().map(|seg| seg.ident.to_string()),
        )),
        Pat::Or(pat) => OpcodeDispatchSelector::Or(
            pat.cases
                .iter()
                .map(extract_opcode_dispatch_selector)
                .collect(),
        ),
        Pat::Wild(_) => OpcodeDispatchSelector::Wildcard,
        _ => OpcodeDispatchSelector::Unsupported,
    }
}

/// Extract handler call identities from an expression.
fn extract_handler_calls(expr: &syn::Expr) -> Vec<ExtractedHandlerCall> {
    let mut calls = Vec::new();
    let mut collector = CallCollector { calls: &mut calls };
    syn::visit::visit_expr(&mut collector, expr);
    calls
}

struct CallCollector<'a> {
    calls: &'a mut Vec<ExtractedHandlerCall>,
}

impl<'ast, 'a> Visit<'ast> for CallCollector<'a> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.calls.push(ExtractedHandlerCall::Method {
            name: call.method.to_string(),
            receiver_root: expr_root_ident(&call.receiver),
        });
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.calls.push(extract_function_call_identity(&call.func));
        syn::visit::visit_expr_call(self, call);
    }
}

fn extract_function_call_identity(expr: &syn::Expr) -> ExtractedHandlerCall {
    match expr {
        syn::Expr::Path(path) => ExtractedHandlerCall::FunctionPath(CallPath::from_segments(
            path.path.segments.iter().map(|seg| seg.ident.to_string()),
        )),
        _ => ExtractedHandlerCall::UnsupportedFunctionExpr,
    }
}

fn expr_root_ident(expr: &syn::Expr) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path.path.get_ident().map(|ident| ident.to_string()),
        syn::Expr::Reference(r) => expr_root_ident(&r.expr),
        syn::Expr::Paren(p) => expr_root_ident(&p.expr),
        syn::Expr::Field(field) => expr_root_ident(&field.base),
        syn::Expr::Index(index) => expr_root_ident(&index.expr),
        _ => None,
    }
}

fn extract_receiver_trait_bindings(func: &ItemFn) -> ReceiverTraitBindings {
    let mut generic_bounds: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for param in &func.sig.generics.params {
        if let syn::GenericParam::Type(ty) = param {
            let bounds = ty
                .bounds
                .iter()
                .filter_map(|bound| match bound {
                    syn::TypeParamBound::Trait(trait_bound) => Some(
                        trait_bound
                            .path
                            .segments
                            .last()
                            .map(|seg| seg.ident.to_string())
                            .unwrap_or_default(),
                    ),
                    _ => None,
                })
                .filter(|name| !name.is_empty())
                .collect::<Vec<_>>();
            if !bounds.is_empty() {
                generic_bounds.insert(ty.ident.to_string(), bounds);
            }
        }
    }

    let mut traits_by_receiver = std::collections::HashMap::new();
    let mut type_root_by_receiver = std::collections::HashMap::new();
    for arg in &func.sig.inputs {
        let syn::FnArg::Typed(arg) = arg else {
            continue;
        };
        let syn::Pat::Ident(pat_ident) = &*arg.pat else {
            continue;
        };
        if let Some(type_name) = type_root_ident(&arg.ty) {
            type_root_by_receiver.insert(pat_ident.ident.to_string(), type_name.clone());
            if let Some(bounds) = generic_bounds.get(&type_name) {
                traits_by_receiver.insert(pat_ident.ident.to_string(), bounds.clone());
            }
        }
    }

    ReceiverTraitBindings {
        traits_by_receiver,
        type_root_by_receiver,
    }
}

/// Returns full type path — all segments joined by "::".
/// RPython: lltype.Struct has globally unique identity.
fn type_root_ident(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => {
            let segments: Vec<_> = path
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            if segments.is_empty() {
                None
            } else {
                Some(segments.join("::"))
            }
        }
        syn::Type::Reference(reference) => type_root_ident(&reference.elem),
        syn::Type::Paren(paren) => type_root_ident(&paren.elem),
        syn::Type::Group(group) => type_root_ident(&group.elem),
        _ => None,
    }
}

fn canonical_path_name(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn canonical_type_name(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(path) => canonical_path_name(&path.path),
        syn::Type::Reference(reference) => canonical_type_name(&reference.elem),
        syn::Type::Paren(paren) => canonical_type_name(&paren.elem),
        syn::Type::Group(group) => canonical_type_name(&group.elem),
        syn::Type::Ptr(ptr) => canonical_type_name(&ptr.elem),
        syn::Type::Slice(slice) => format!("[{}]", canonical_type_name(&slice.elem)),
        syn::Type::Array(array) => format!("[{}]", canonical_type_name(&array.elem)),
        _ => "<unsupported-type>".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_function_call_identity_preserves_path_segments() {
        let expr: syn::Expr = syn::parse_quote!(crate::runtime::exec_build_list(frame, 1));
        let call = match expr {
            syn::Expr::Call(call) => call,
            _ => panic!("expected call expr"),
        };

        let identity = extract_function_call_identity(&call.func);
        assert_eq!(
            identity,
            ExtractedHandlerCall::FunctionPath(CallPath::from_segments([
                "crate",
                "runtime",
                "exec_build_list",
            ]))
        );
    }

    #[test]
    fn extract_receiver_trait_bindings_from_execute_opcode_step() {
        let parsed = parse_source(
            r#"
            pub trait OpcodeStepExecutor { fn load_fast_checked(&mut self, idx: usize); }
            pub fn execute_opcode_step<E: OpcodeStepExecutor>(executor: &mut E, idx: usize) {
                executor.load_fast_checked(idx);
            }
        "#,
        );

        let bindings = extract_opcode_dispatch_receiver_traits(&parsed);
        assert_eq!(
            bindings.traits_by_receiver.get("executor"),
            Some(&vec!["OpcodeStepExecutor".to_string()])
        );
    }

    #[test]
    fn extract_trait_default_methods_include_graphs() {
        let parsed = parse_source(
            r#"
            trait Foo {
                fn helper(&mut self, x: i64) -> i64 {
                    x + 1
                }
            }
        "#,
        );
        let impls = extract_trait_impls(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
        );
        let helper = impls[0]
            .methods
            .iter()
            .find(|m| m.name == "helper")
            .expect("helper method");
        assert!(
            helper.graph.is_some(),
            "trait default method should carry graph"
        );
    }

    #[test]
    fn extract_opcode_dispatch_selector_uses_exact_variant_path() {
        let pat: syn::Pat = syn::parse_quote!(Instruction::LoadFast { var_num });
        let selector = extract_opcode_dispatch_selector(&pat);
        assert_eq!(
            selector,
            OpcodeDispatchSelector::Path(CallPath::from_segments(["Instruction", "LoadFast",]))
        );
        assert_eq!(selector.canonical_key(), "Instruction::LoadFast");
    }

    #[test]
    fn extract_opcode_dispatch_selector_preserves_or_cases() {
        let pat: syn::Pat = syn::parse_quote!(Instruction::LoadFast | Instruction::StoreFast);
        let selector = extract_opcode_dispatch_selector(&pat);
        assert_eq!(
            selector.canonical_key(),
            "Instruction::LoadFast | Instruction::StoreFast"
        );
    }

    #[test]
    fn find_function_uses_canonical_parse_surface() {
        let parsed = parse_source(
            r#"
            fn helper() {}
            fn mainloop() {}
        "#,
        );
        let func = find_function(&parsed, "mainloop").expect("mainloop");
        assert_eq!(func.sig.ident, "mainloop");
    }

    #[test]
    fn find_opcode_match_uses_canonical_parse_surface() {
        let parsed = parse_source(
            r#"
            fn mainloop() {
                match op {
                    OP_ADD => {},
                    _ => {},
                }
            }
        "#,
        );
        let func = find_function(&parsed, "mainloop").expect("mainloop");
        let opcode_match = find_opcode_match(func).expect("opcode match");
        assert_eq!(opcode_match.arms.len(), 2);
    }

    #[test]
    fn find_opcode_match_finds_nested_dispatch() {
        let parsed = parse_source(
            r#"
            fn mainloop() {
                loop {
                    match op {
                        OP_ADD => {},
                        OP_SUB => {},
                        _ => {},
                    }
                }
            }
        "#,
        );
        let func = find_function(&parsed, "mainloop").expect("mainloop");
        let opcode_match = find_opcode_match(func).expect("opcode match");
        assert_eq!(opcode_match.arms.len(), 3);
    }

    #[test]
    fn find_opcode_match_accepts_instruction_enum_dispatch() {
        let parsed = parse_source(
            r#"
            fn execute_opcode_step(inst: Instruction) {
                match inst {
                    Instruction::LoadConst { idx } => {}
                    Instruction::Add => {}
                    _ => {}
                }
            }
        "#,
        );
        let func = find_function(&parsed, "execute_opcode_step").expect("execute_opcode_step");
        let opcode_match = find_opcode_match(func).expect("opcode match");
        assert_eq!(opcode_match.arms.len(), 3);
    }
}
