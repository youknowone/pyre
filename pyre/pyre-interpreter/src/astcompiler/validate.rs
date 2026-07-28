//! A visitor to validate an AST object.
//!
//! `compile()` given a tree of `_ast` objects converts it and then walks the
//! result here before handing it to the code generator, so a tree the parser
//! could never have produced is rejected with a message rather than compiled.
//!
//! The walk is over the compiler AST, which cannot hold every shape the
//! objects can: a list element that is missing, a `Dict` whose keys and values
//! have different lengths, defaults that outnumber their parameters. Those are
//! rejected by the conversion in `module::_ast::convert`, which is the last
//! place they are still visible; everything else is checked here.

use rustpython_compiler::ast;

use crate::PyError;

type ValidateResult = Result<(), PyError>;

/// Seen as a ValueError.
fn validation_error(message: impl Into<String>) -> PyError {
    PyError::value_error(message)
}

/// Seen as a TypeError.
fn validation_type_error(message: impl Into<String>) -> PyError {
    PyError::type_error(message)
}

pub fn validate_ast(node: &ast::Mod) -> ValidateResult {
    let validator = AstValidator;
    match node {
        ast::Mod::Module(node) => validator.validate_stmts(&node.body),
        ast::Mod::Expression(node) => validator.validate_expr(&node.body, ast::ExprContext::Load),
    }
}

fn expr_context_name(ctx: ast::ExprContext) -> &'static str {
    match ctx {
        ast::ExprContext::Load => "Load",
        ast::ExprContext::Store => "Store",
        ast::ExprContext::Del => "Del",
        ast::ExprContext::Invalid => "??",
    }
}

fn check_context(expected_ctx: ast::ExprContext, actual_ctx: ast::ExprContext) -> ValidateResult {
    if expected_ctx != actual_ctx {
        return Err(validation_error(format!(
            "expression must have {} context but has {} instead",
            expr_context_name(expected_ctx),
            expr_context_name(actual_ctx)
        )));
    }
    Ok(())
}

/// The context an expression carries, for the ones that carry one.  The rest
/// cannot be assigned to at all.
fn context_of(expr: &ast::Expr) -> Option<ast::ExprContext> {
    match expr {
        ast::Expr::Name(node) => Some(node.ctx),
        ast::Expr::List(node) => Some(node.ctx),
        ast::Expr::Tuple(node) => Some(node.ctx),
        ast::Expr::Starred(node) => Some(node.ctx),
        ast::Expr::Subscript(node) => Some(node.ctx),
        ast::Expr::Attribute(node) => Some(node.ctx),
        _ => None,
    }
}

/// Recursive function to validate a Constant value.
fn validate_constant(value: &ast::ConstantValue) -> ValidateResult {
    match value {
        ast::ConstantValue::None
        | ast::ConstantValue::Ellipsis
        | ast::ConstantValue::Boolean(_)
        | ast::ConstantValue::Integer(_)
        | ast::ConstantValue::Float(_)
        | ast::ConstantValue::Complex { .. }
        | ast::ConstantValue::Str(_)
        | ast::ConstantValue::Bytes(_) => Ok(()),
        ast::ConstantValue::Tuple(items) | ast::ConstantValue::Frozenset(items) => {
            items.iter().try_for_each(validate_constant)
        }
    }
}

struct AstValidator;

impl AstValidator {
    fn validate_stmts(&self, stmts: &[ast::Stmt]) -> ValidateResult {
        stmts.iter().try_for_each(|stmt| self.visit_stmt(stmt))
    }

    fn validate_expr(&self, expr: &ast::Expr, ctx: ast::ExprContext) -> ValidateResult {
        match context_of(expr) {
            Some(actual) => check_context(ctx, actual)?,
            None if ctx != ast::ExprContext::Load => {
                return Err(validation_error(format!(
                    "expression which can't be assigned to in {} context",
                    expr_context_name(ctx)
                )));
            }
            None => {}
        }
        self.walkabout_with_ctx(expr, ctx)
    }

    /// `List`, `Tuple` and `Starred` pass their context down to what they
    /// hold; every other expression is entered in Load context.
    fn walkabout_with_ctx(&self, expr: &ast::Expr, ctx: ast::ExprContext) -> ValidateResult {
        match expr {
            ast::Expr::List(node) => self.validate_exprs(&node.elts, ctx),
            ast::Expr::Tuple(node) => self.validate_exprs(&node.elts, ctx),
            ast::Expr::Starred(node) => self.validate_expr(&node.value, ctx),
            _ => self.visit_expr(expr),
        }
    }

    /// Upstream takes a `null_ok` for the two lists that may hold a missing
    /// element -- `Dict` keys and `kw_defaults`.  Neither survives into the
    /// compiler AST as a list with holes, so there is nothing here to allow.
    fn validate_exprs(&self, exprs: &[ast::Expr], ctx: ast::ExprContext) -> ValidateResult {
        exprs
            .iter()
            .try_for_each(|expr| self.validate_expr(expr, ctx))
    }

    fn validate_body(&self, body: &[ast::Stmt], owner: &str) -> ValidateResult {
        self.validate_nonempty_seq(body.len(), "body", owner)?;
        self.validate_stmts(body)
    }

    fn validate_nonempty_seq(&self, len: usize, what: &str, owner: &str) -> ValidateResult {
        if len == 0 {
            return Err(validation_error(format!("empty {what} on {owner}")));
        }
        Ok(())
    }

    fn validate_name(&self, name: &str) -> ValidateResult {
        if matches!(name, "None" | "True" | "False") {
            return Err(validation_error(format!(
                "identifier field can't represent '{name}' constant"
            )));
        }
        Ok(())
    }

    // Statements

    fn visit_stmt(&self, stmt: &ast::Stmt) -> ValidateResult {
        match stmt {
            ast::Stmt::FunctionDef(node) => self.visit_function_def(node),
            ast::Stmt::ClassDef(node) => self.visit_class_def(node),
            ast::Stmt::Return(node) => match &node.value {
                Some(value) => self.validate_expr(value, ast::ExprContext::Load),
                None => Ok(()),
            },
            ast::Stmt::Delete(node) => {
                self.validate_nonempty_seq(node.targets.len(), "targets", "Delete")?;
                self.validate_exprs(&node.targets, ast::ExprContext::Del)
            }
            ast::Stmt::Assign(node) => {
                self.validate_nonempty_seq(node.targets.len(), "targets", "Assign")?;
                self.validate_exprs(&node.targets, ast::ExprContext::Store)?;
                self.validate_expr(&node.value, ast::ExprContext::Load)
            }
            ast::Stmt::AugAssign(node) => {
                self.validate_expr(&node.target, ast::ExprContext::Store)?;
                self.validate_expr(&node.value, ast::ExprContext::Load)
            }
            ast::Stmt::AnnAssign(node) => {
                self.validate_expr(&node.target, ast::ExprContext::Store)?;
                self.validate_expr(&node.annotation, ast::ExprContext::Load)?;
                match &node.value {
                    Some(value) => self.validate_expr(value, ast::ExprContext::Load),
                    None => Ok(()),
                }
            }
            ast::Stmt::TypeAlias(node) => {
                self.validate_expr(&node.name, ast::ExprContext::Store)?;
                self.validate_expr(&node.value, ast::ExprContext::Load)
            }
            ast::Stmt::For(node) => {
                let owner = if node.is_async { "AsyncFor" } else { "For" };
                self.validate_expr(&node.target, ast::ExprContext::Store)?;
                self.validate_expr(&node.iter, ast::ExprContext::Load)?;
                self.validate_body(&node.body, owner)?;
                self.validate_stmts(&node.orelse)
            }
            ast::Stmt::While(node) => {
                self.validate_expr(&node.test, ast::ExprContext::Load)?;
                self.validate_body(&node.body, "While")?;
                self.validate_stmts(&node.orelse)
            }
            ast::Stmt::If(node) => {
                self.validate_expr(&node.test, ast::ExprContext::Load)?;
                self.validate_body(&node.body, "If")?;
                // An `elif` is the `orelse` holding one more `If`, so its own
                // body carries the same requirement; a plain `else` is only a
                // statement list.
                for clause in &node.elif_else_clauses {
                    match &clause.test {
                        Some(test) => {
                            self.validate_expr(test, ast::ExprContext::Load)?;
                            self.validate_body(&clause.body, "If")?;
                        }
                        None => self.validate_stmts(&clause.body)?,
                    }
                }
                Ok(())
            }
            ast::Stmt::With(node) => {
                let owner = if node.is_async { "AsyncWith" } else { "With" };
                self.validate_nonempty_seq(node.items.len(), "items", owner)?;
                for item in &node.items {
                    self.validate_expr(&item.context_expr, ast::ExprContext::Load)?;
                    if let Some(vars) = item.optional_vars.as_deref() {
                        self.validate_expr(vars, ast::ExprContext::Store)?;
                    }
                }
                self.validate_body(&node.body, owner)
            }
            ast::Stmt::Raise(node) => match (&node.exc, &node.cause) {
                (Some(exc), cause) => {
                    self.validate_expr(exc, ast::ExprContext::Load)?;
                    match cause {
                        Some(cause) => self.validate_expr(cause, ast::ExprContext::Load),
                        None => Ok(()),
                    }
                }
                (None, Some(_)) => Err(validation_error("Raise with cause but no exception")),
                (None, None) => Ok(()),
            },
            ast::Stmt::Try(node) => self.visit_try(node),
            ast::Stmt::Assert(node) => {
                self.validate_expr(&node.test, ast::ExprContext::Load)?;
                match &node.msg {
                    Some(msg) => self.validate_expr(msg, ast::ExprContext::Load),
                    None => Ok(()),
                }
            }
            ast::Stmt::Import(node) => {
                self.validate_nonempty_seq(node.names.len(), "names", "Import")
            }
            ast::Stmt::ImportFrom(node) => {
                self.validate_nonempty_seq(node.names.len(), "names", "ImportFrom")
            }
            ast::Stmt::Global(node) => {
                self.validate_nonempty_seq(node.names.len(), "names", "Global")
            }
            ast::Stmt::Nonlocal(node) => {
                self.validate_nonempty_seq(node.names.len(), "names", "Nonlocal")
            }
            ast::Stmt::Expr(node) => self.validate_expr(&node.value, ast::ExprContext::Load),
            ast::Stmt::Match(node) => self.visit_match(node),
            ast::Stmt::Pass(_) | ast::Stmt::Break(_) | ast::Stmt::Continue(_) => Ok(()),
            ast::Stmt::IpyEscapeCommand(_) => Ok(()),
        }
    }

    fn visit_function_def(&self, node: &ast::StmtFunctionDef) -> ValidateResult {
        let owner = if node.is_async {
            "AsyncFunctionDef"
        } else {
            "FunctionDef"
        };
        self.validate_body(&node.body, owner)?;
        self.visit_parameters(&node.parameters)?;
        self.validate_decorators(&node.decorator_list)?;
        match &node.returns {
            Some(returns) => self.validate_expr(returns, ast::ExprContext::Load),
            None => Ok(()),
        }
    }

    fn visit_class_def(&self, node: &ast::StmtClassDef) -> ValidateResult {
        self.validate_body(&node.body, "ClassDef")?;
        if let Some(arguments) = node.arguments.as_deref() {
            self.validate_exprs(&arguments.args, ast::ExprContext::Load)?;
            for keyword in &arguments.keywords {
                self.validate_expr(&keyword.value, ast::ExprContext::Load)?;
            }
        }
        self.validate_decorators(&node.decorator_list)
    }

    fn validate_decorators(&self, decorators: &[ast::Decorator]) -> ValidateResult {
        decorators.iter().try_for_each(|decorator| {
            self.validate_expr(&decorator.expression, ast::ExprContext::Load)
        })
    }

    fn visit_parameters(&self, node: &ast::Parameters) -> ValidateResult {
        for parameter in node
            .posonlyargs
            .iter()
            .chain(&node.args)
            .chain(&node.kwonlyargs)
        {
            self.visit_parameter(&parameter.parameter)?;
            if let Some(default) = parameter.default.as_deref() {
                self.validate_expr(default, ast::ExprContext::Load)?;
            }
        }
        for parameter in [node.vararg.as_deref(), node.kwarg.as_deref()]
            .into_iter()
            .flatten()
        {
            self.visit_parameter(parameter)?;
        }
        Ok(())
    }

    fn visit_parameter(&self, node: &ast::Parameter) -> ValidateResult {
        match &node.annotation {
            Some(annotation) => self.validate_expr(annotation, ast::ExprContext::Load),
            None => Ok(()),
        }
    }

    fn visit_try(&self, node: &ast::StmtTry) -> ValidateResult {
        let owner = if node.is_star { "TryStar" } else { "Try" };
        self.validate_body(&node.body, owner)?;
        if node.handlers.is_empty() && node.finalbody.is_empty() {
            return Err(validation_error(format!(
                "{owner} has neither except handlers nor finalbody"
            )));
        }
        if node.handlers.is_empty() && !node.orelse.is_empty() {
            return Err(validation_error(format!(
                "{owner} has orelse but no except handlers"
            )));
        }
        for handler in &node.handlers {
            let ast::ExceptHandler::ExceptHandler(handler) = handler;
            if let Some(type_) = handler.type_.as_deref() {
                self.validate_expr(type_, ast::ExprContext::Load)?;
            }
            self.validate_body(&handler.body, "ExceptHandler")?;
        }
        self.validate_stmts(&node.orelse)?;
        self.validate_stmts(&node.finalbody)
    }

    // pattern matching

    /// A second star in the same sequence is left to the code generator, which
    /// reports it where 3.14 does, as a SyntaxError; upstream raises here and
    /// so answers with a ValueError.
    fn validate_patterns(&self, patterns: &[ast::Pattern], star_ok: bool) -> ValidateResult {
        for pattern in patterns {
            if let ast::Pattern::MatchStar(node) = pattern
                && star_ok
            {
                if let Some(name) = &node.name {
                    self.validate_capture(name)?;
                }
                continue;
            }
            self.visit_pattern(pattern)?;
        }
        Ok(())
    }

    fn validate_capture(&self, name: &str) -> ValidateResult {
        if name == "_" {
            return Err(validation_error("can't capture name '_' in patterns"));
        }
        self.validate_name(name)
    }

    fn visit_match(&self, node: &ast::StmtMatch) -> ValidateResult {
        self.validate_nonempty_seq(node.cases.len(), "cases", "Match")?;
        self.validate_expr(&node.subject, ast::ExprContext::Load)?;
        for case in &node.cases {
            if let Some(guard) = case.guard.as_deref() {
                self.validate_expr(guard, ast::ExprContext::Load)?;
            }
            self.validate_stmts(&case.body)?;
            self.visit_pattern(&case.pattern)?;
        }
        Ok(())
    }

    fn visit_pattern(&self, pattern: &ast::Pattern) -> ValidateResult {
        match pattern {
            ast::Pattern::MatchValue(node) => {
                self.validate_expr(&node.value, ast::ExprContext::Load)?;
                let literal = match node.value.as_ref() {
                    ast::Expr::Constant(constant) => matches!(
                        constant.value,
                        ast::ConstantValue::Integer(_)
                            | ast::ConstantValue::Float(_)
                            | ast::ConstantValue::Complex { .. }
                            | ast::ConstantValue::Str(_)
                            | ast::ConstantValue::Bytes(_)
                    ),
                    // Anything the parser produced is a literal or an
                    // attribute lookup already.
                    _ => true,
                };
                if literal {
                    Ok(())
                } else {
                    Err(validation_error(
                        "unexpected constant inside of a literal pattern",
                    ))
                }
            }
            ast::Pattern::MatchSingleton(_) => Ok(()),
            ast::Pattern::MatchSequence(node) => self.validate_patterns(&node.patterns, true),
            ast::Pattern::MatchMapping(node) => {
                if !node.keys.is_empty()
                    && !node.patterns.is_empty()
                    && node.keys.len() != node.patterns.len()
                {
                    return Err(validation_error(
                        "MatchMapping doesn't have the same number of keys as patterns",
                    ));
                }
                for key in &node.keys {
                    if !matches!(key, ast::Expr::Constant(_) | ast::Expr::Attribute(_)) {
                        // Upstream words this as "can only have Constant or
                        // Attribute in the keys of a MatchMapping".
                        return Err(validation_error(
                            "patterns may only match literals and attribute lookups",
                        ));
                    }
                }
                if let Some(rest) = &node.rest {
                    self.validate_capture(rest)?;
                }
                self.validate_patterns(&node.patterns, false)
            }
            ast::Pattern::MatchClass(node) => {
                self.validate_expr(&node.cls, ast::ExprContext::Load)?;
                let mut cls = node.cls.as_ref();
                loop {
                    match cls {
                        ast::Expr::Name(_) => break,
                        ast::Expr::Attribute(attribute) => cls = &attribute.value,
                        _ => {
                            return Err(validation_error(
                                "MatchClass cls field can only contain Name or Attribute nodes.",
                            ));
                        }
                    }
                }
                for keyword in &node.arguments.keywords {
                    self.validate_name(&keyword.attr)?;
                }
                self.validate_patterns(&node.arguments.patterns, false)?;
                for keyword in &node.arguments.keywords {
                    self.visit_pattern(&keyword.pattern)?;
                }
                Ok(())
            }
            ast::Pattern::MatchStar(_) => Err(validation_error("can't use MatchStar here")),
            ast::Pattern::MatchAs(node) => {
                if let Some(name) = &node.name {
                    self.validate_capture(name)?;
                }
                match (&node.pattern, &node.name) {
                    (Some(_), None) => Err(validation_error(
                        "MatchAs must specify a target name if a pattern is given",
                    )),
                    (Some(pattern), Some(_)) => self.visit_pattern(pattern),
                    (None, _) => Ok(()),
                }
            }
            ast::Pattern::MatchOr(node) => {
                if node.patterns.len() < 2 {
                    return Err(validation_error("MatchOr requires at least 2 patterns"));
                }
                self.validate_patterns(&node.patterns, false)
            }
        }
    }

    // Expressions

    fn visit_expr(&self, expr: &ast::Expr) -> ValidateResult {
        match expr {
            ast::Expr::Name(node) => self.validate_name(&node.id),
            ast::Expr::Constant(node) => validate_constant(&node.value),
            ast::Expr::BoolOp(node) => {
                if node.values.len() < 2 {
                    return Err(validation_error("BoolOp with less than 2 values"));
                }
                self.validate_exprs(&node.values, ast::ExprContext::Load)
            }
            ast::Expr::UnaryOp(node) => self.validate_expr(&node.operand, ast::ExprContext::Load),
            ast::Expr::BinOp(node) => {
                self.validate_expr(&node.left, ast::ExprContext::Load)?;
                self.validate_expr(&node.right, ast::ExprContext::Load)
            }
            ast::Expr::Lambda(node) => {
                if let Some(parameters) = node.parameters.as_deref() {
                    self.visit_parameters(parameters)?;
                }
                self.validate_expr(&node.body, ast::ExprContext::Load)
            }
            ast::Expr::If(node) => {
                self.validate_expr(&node.test, ast::ExprContext::Load)?;
                self.validate_expr(&node.body, ast::ExprContext::Load)?;
                self.validate_expr(&node.orelse, ast::ExprContext::Load)
            }
            ast::Expr::Dict(node) => {
                for item in &node.items {
                    if let Some(key) = &item.key {
                        self.validate_expr(key, ast::ExprContext::Load)?;
                    }
                    self.validate_expr(&item.value, ast::ExprContext::Load)?;
                }
                Ok(())
            }
            ast::Expr::Set(node) => self.validate_exprs(&node.elts, ast::ExprContext::Load),
            ast::Expr::ListComp(node) => {
                self.validate_comprehension(&node.generators)?;
                self.validate_expr(&node.elt, ast::ExprContext::Load)
            }
            ast::Expr::SetComp(node) => {
                self.validate_comprehension(&node.generators)?;
                self.validate_expr(&node.elt, ast::ExprContext::Load)
            }
            ast::Expr::Generator(node) => {
                self.validate_comprehension(&node.generators)?;
                self.validate_expr(&node.elt, ast::ExprContext::Load)
            }
            ast::Expr::DictComp(node) => {
                self.validate_comprehension(&node.generators)?;
                self.validate_expr(&node.key, ast::ExprContext::Load)?;
                self.validate_expr(&node.value, ast::ExprContext::Load)
            }
            ast::Expr::Await(node) => self.validate_expr(&node.value, ast::ExprContext::Load),
            ast::Expr::Yield(node) => match &node.value {
                Some(value) => self.validate_expr(value, ast::ExprContext::Load),
                None => Ok(()),
            },
            ast::Expr::YieldFrom(node) => self.validate_expr(&node.value, ast::ExprContext::Load),
            ast::Expr::Compare(node) => {
                if node.comparators.is_empty() {
                    return Err(validation_error("Compare with no comparators"));
                }
                if node.comparators.len() != node.ops.len() {
                    return Err(validation_error(
                        "Compare has a different number of comparators and operands",
                    ));
                }
                self.validate_exprs(&node.comparators, ast::ExprContext::Load)?;
                self.validate_expr(&node.left, ast::ExprContext::Load)
            }
            ast::Expr::Call(node) => {
                self.validate_expr(&node.func, ast::ExprContext::Load)?;
                self.validate_exprs(&node.arguments.args, ast::ExprContext::Load)?;
                for keyword in &node.arguments.keywords {
                    self.validate_expr(&keyword.value, ast::ExprContext::Load)?;
                }
                Ok(())
            }
            ast::Expr::Attribute(node) => self.validate_expr(&node.value, ast::ExprContext::Load),
            ast::Expr::Subscript(node) => {
                self.validate_expr(&node.value, ast::ExprContext::Load)?;
                self.validate_expr(&node.slice, ast::ExprContext::Load)
            }
            ast::Expr::Slice(node) => {
                for part in [
                    node.lower.as_deref(),
                    node.upper.as_deref(),
                    node.step.as_deref(),
                ]
                .into_iter()
                .flatten()
                {
                    self.validate_expr(part, ast::ExprContext::Load)?;
                }
                Ok(())
            }
            ast::Expr::FString(node) => self.visit_fstring(node),
            ast::Expr::Named(node) => {
                if !matches!(node.target.as_ref(), ast::Expr::Name(_)) {
                    return Err(validation_type_error("NamedExpr target must be a Name"));
                }
                self.validate_expr(&node.value, ast::ExprContext::Load)
            }
            ast::Expr::Starred(node) => self.validate_expr(&node.value, ast::ExprContext::Load),
            ast::Expr::List(node) => self.validate_exprs(&node.elts, ast::ExprContext::Load),
            ast::Expr::Tuple(node) => self.validate_exprs(&node.elts, ast::ExprContext::Load),
            // A literal the parser produced, or a node with nothing to check.
            ast::Expr::StringLiteral(_)
            | ast::Expr::BytesLiteral(_)
            | ast::Expr::NumberLiteral(_)
            | ast::Expr::BooleanLiteral(_)
            | ast::Expr::NoneLiteral(_)
            | ast::Expr::EllipsisLiteral(_)
            | ast::Expr::TString(_)
            | ast::Expr::IpyEscapeCommand(_) => Ok(()),
        }
    }

    /// The values of a `JoinedStr` reach the compiler AST as the parts to
    /// join, and a `FormattedValue` spec as the expression to format with.
    fn visit_fstring(&self, node: &ast::ExprFString) -> ValidateResult {
        if let Some(values) = node.runtime_joined_str.as_deref() {
            return self.validate_exprs(values, ast::ExprContext::Load);
        }
        for element in node.value.elements() {
            if let ast::InterpolatedStringElement::Interpolation(interpolation) = element {
                self.validate_expr(&interpolation.expression, ast::ExprContext::Load)?;
                if let Some(spec) = interpolation.runtime_formatted_value_format_spec.as_deref() {
                    self.validate_expr(spec, ast::ExprContext::Load)?;
                }
            }
        }
        Ok(())
    }

    fn validate_comprehension(&self, generators: &[ast::Comprehension]) -> ValidateResult {
        if generators.is_empty() {
            return Err(validation_error("comprehension with no generators"));
        }
        for comprehension in generators {
            self.validate_expr(&comprehension.target, ast::ExprContext::Store)?;
            self.validate_expr(&comprehension.iter, ast::ExprContext::Load)?;
            self.validate_exprs(&comprehension.ifs, ast::ExprContext::Load)?;
        }
        Ok(())
    }
}
