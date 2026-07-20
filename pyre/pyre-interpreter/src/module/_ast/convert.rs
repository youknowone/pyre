//! RustPython/Ruff AST → interpreter-level `_ast` objects.
//!
//! PyPy's `ast.Node.to_object(space)` performs this same boundary conversion:
//! parser nodes stay native to the compiler, while the public `ast` module sees
//! ordinary heap objects carrying ASDL fields and source locations.

use pyre_object::{PY_NULL, PyObjectRef};
use rustpython_compiler::{ast, parser};

type AstResult<T> = Result<T, crate::PyError>;

/// Convert an interpreter-level `_ast` tree back into Ruff's compiler AST and
/// compile it.  This is the reverse of `Converter`, corresponding to PyPy's
/// generated `ast_from_object` boundary.
pub fn compile_object(
    object: PyObjectRef,
    filename: &str,
    mode: crate::compile::Mode,
    opts: crate::compile::CompileOpts,
) -> AstResult<crate::compile::CodeObject> {
    let ast_module = crate::importing::importhook(
        "_ast",
        PY_NULL,
        PY_NULL,
        0,
        crate::call::take_last_exec_ctx(),
    )?;
    let mut converter = ObjectConverter {
        ast_module,
        depth: 0,
    };
    let module = converter.module(object)?;
    let source_file = rustpython_compiler::core::SourceFileBuilder::new(filename, "").finish();
    rustpython_compiler::codegen::compile::compile_top(module, source_file, mode, opts)
        .map_err(|error| crate::PyError::syntax_error(error.to_string()))
}

struct ObjectConverter {
    ast_module: PyObjectRef,
    depth: usize,
}

impl ObjectConverter {
    fn recurse<T>(&mut self, f: impl FnOnce(&mut Self) -> AstResult<T>) -> AstResult<T> {
        // PyPy's generated ast_from_object calls space.getexecutioncontext()
        // recursion guards around nested ASDL nodes.  Keep the state on this
        // conversion, never in TLS or a global side table.
        if self.depth >= 200 {
            return Err(crate::PyError::recursion_error(
                "maximum recursion depth exceeded while traversing AST node",
            ));
        }
        self.depth += 1;
        let result = f(self);
        self.depth -= 1;
        result
    }

    fn is_node(&self, object: PyObjectRef, name: &str) -> AstResult<bool> {
        let ty = crate::baseobjspace::getattr_str(self.ast_module, name)?;
        Ok(unsafe { crate::baseobjspace::isinstance_w(object, ty) })
    }

    fn field(&self, object: PyObjectRef, field: &str, node: &str) -> AstResult<PyObjectRef> {
        crate::baseobjspace::getattr_str(object, field).map_err(|_| {
            crate::PyError::type_error(format!("required field {field:?} missing from {node}"))
        })
    }

    fn optional_field(&self, object: PyObjectRef, field: &str) -> AstResult<Option<PyObjectRef>> {
        match crate::baseobjspace::getattr_str(object, field) {
            Ok(value) if unsafe { pyre_object::is_none(value) } => Ok(None),
            Ok(value) => Ok(Some(value)),
            Err(error) if error.kind == crate::PyErrorKind::AttributeError => Ok(None),
            Err(error) => Err(error),
        }
    }

    fn list(&self, object: PyObjectRef, field: &str, node: &str) -> AstResult<Vec<PyObjectRef>> {
        let value = self.field(object, field, node)?;
        if !unsafe { pyre_object::is_list(value) } {
            return Err(crate::PyError::type_error(format!(
                "AST list field must be a list, not {}",
                unsafe { pyre_object::type_name_of(value) }
            )));
        }
        Ok(unsafe { pyre_object::w_list_items_copy_as_vec(value) })
    }

    fn string(&self, object: PyObjectRef, field: &str, node: &str) -> AstResult<String> {
        let value = self.field(object, field, node)?;
        if !unsafe { pyre_object::is_str(value) } {
            return Err(crate::PyError::type_error(
                "AST identifier must be of type str",
            ));
        }
        Ok(unsafe { pyre_object::w_str_get_value(value).to_string() })
    }

    fn module(&mut self, object: PyObjectRef) -> AstResult<ast::Mod> {
        if self.is_node(object, "Module")? {
            let values = self.list(object, "body", "Module")?;
            let body = values
                .into_iter()
                .map(|value| self.recurse(|this| this.stmt(value)))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ast::Mod::Module(ast::ModModule {
                node_index: Default::default(),
                range: Default::default(),
                body,
                runtime_body: None,
            }))
        } else if self.is_node(object, "Interactive")? {
            let values = self.list(object, "body", "Interactive")?;
            let body = values
                .into_iter()
                .map(|value| self.recurse(|this| this.stmt(value)))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ast::Mod::Module(ast::ModModule {
                node_index: Default::default(),
                range: Default::default(),
                body,
                runtime_body: None,
            }))
        } else if self.is_node(object, "Expression")? {
            let body = self.field(object, "body", "Expression")?;
            Ok(ast::Mod::Expression(ast::ModExpression {
                node_index: Default::default(),
                range: Default::default(),
                body: Box::new(self.recurse(|this| this.expr(body))?),
            }))
        } else {
            Err(crate::PyError::type_error(format!(
                "expected some sort of mod, but got {}",
                unsafe { pyre_object::type_name_of(object) }
            )))
        }
    }

    fn stmt(&mut self, object: PyObjectRef) -> AstResult<ast::Stmt> {
        if self.is_node(object, "FunctionDef")? || self.is_node(object, "AsyncFunctionDef")? {
            let is_async = self.is_node(object, "AsyncFunctionDef")?;
            let name = ast::Identifier::new(
                self.string(
                    object,
                    "name",
                    if is_async {
                        "AsyncFunctionDef"
                    } else {
                        "FunctionDef"
                    },
                )?,
                Default::default(),
            );
            let args = self.field(object, "args", "FunctionDef")?;
            let parameters = Box::new(self.recurse(|this| this.parameters(args))?);
            let body = self
                .list(object, "body", "FunctionDef")?
                .into_iter()
                .map(|value| self.recurse(|this| this.stmt(value)))
                .collect::<Result<Vec<_>, _>>()?;
            // `type_params` was added after the original positional
            // FunctionDef constructor.  Like RustPython/PyPy, a missing field
            // on a manually constructed legacy node means an empty list.
            let _type_params = self.optional_field(object, "type_params")?;
            Ok(ast::Stmt::FunctionDef(ast::StmtFunctionDef {
                node_index: Default::default(),
                range: Default::default(),
                is_async,
                decorator_list: Vec::new(),
                name,
                type_params: None,
                parameters,
                returns: None,
                body,
                runtime_decorator_list: None,
                runtime_type_comment: None,
                runtime_type_comment_bytes: None,
                runtime_body: None,
            }))
        } else if self.is_node(object, "Pass")? {
            Ok(ast::Stmt::Pass(ast::StmtPass {
                node_index: Default::default(),
                range: Default::default(),
            }))
        } else if self.is_node(object, "Expr")? {
            let value = self.field(object, "value", "Expr")?;
            Ok(ast::Stmt::Expr(ast::StmtExpr {
                node_index: Default::default(),
                range: Default::default(),
                value: Box::new(self.recurse(|this| this.expr(value))?),
            }))
        } else if self.is_node(object, "Return")? {
            let value = self.optional_field(object, "value")?;
            Ok(ast::Stmt::Return(ast::StmtReturn {
                node_index: Default::default(),
                range: Default::default(),
                value: value
                    .map(|value| self.recurse(|this| this.expr(value)).map(Box::new))
                    .transpose()?,
            }))
        } else if self.is_node(object, "Assign")? {
            let targets = self
                .list(object, "targets", "Assign")?
                .into_iter()
                .map(|value| self.recurse(|this| this.expr(value)))
                .collect::<Result<Vec<_>, _>>()?;
            let value = self.field(object, "value", "Assign")?;
            Ok(ast::Stmt::Assign(ast::StmtAssign {
                node_index: Default::default(),
                range: Default::default(),
                targets,
                value: Box::new(self.recurse(|this| this.expr(value))?),
                runtime_targets: None,
                runtime_type_comment: None,
                runtime_type_comment_bytes: None,
            }))
        } else {
            Err(crate::PyError::type_error(format!(
                "expected some sort of stmt, but got {}",
                unsafe { pyre_object::type_name_of(object) }
            )))
        }
    }

    fn parameters(&mut self, object: PyObjectRef) -> AstResult<ast::Parameters> {
        // Preserve the ASDL field reads even for empty argument lists.  Defaults
        // are paired with parameters in source order by the complete converter.
        for field in [
            "posonlyargs",
            "args",
            "kwonlyargs",
            "kw_defaults",
            "defaults",
        ] {
            if !self.list(object, field, "arguments")?.is_empty() {
                return Err(crate::PyError::not_implemented(
                    "compiling AST functions with parameters is not implemented",
                ));
            }
        }
        if self.optional_field(object, "vararg")?.is_some()
            || self.optional_field(object, "kwarg")?.is_some()
        {
            return Err(crate::PyError::not_implemented(
                "compiling AST functions with variadic parameters is not implemented",
            ));
        }
        Ok(ast::Parameters::default())
    }

    fn expr(&mut self, object: PyObjectRef) -> AstResult<ast::Expr> {
        if self.is_node(object, "UnaryOp")? {
            let operand = self.field(object, "operand", "UnaryOp")?;
            let op = self.field(object, "op", "UnaryOp")?;
            Ok(ast::Expr::UnaryOp(ast::ExprUnaryOp {
                node_index: Default::default(),
                range: Default::default(),
                op: self.unaryop(op)?,
                operand: Box::new(self.recurse(|this| this.expr(operand))?),
            }))
        } else if self.is_node(object, "BinOp")? {
            let left = self.field(object, "left", "BinOp")?;
            let right = self.field(object, "right", "BinOp")?;
            let op = self.field(object, "op", "BinOp")?;
            Ok(ast::Expr::BinOp(ast::ExprBinOp {
                node_index: Default::default(),
                range: Default::default(),
                left: Box::new(self.recurse(|this| this.expr(left))?),
                op: self.operator(op)?,
                right: Box::new(self.recurse(|this| this.expr(right))?),
            }))
        } else if self.is_node(object, "Call")? {
            let func = self.field(object, "func", "Call")?;
            let args = self
                .list(object, "args", "Call")?
                .into_iter()
                .map(|arg| self.recurse(|this| this.expr(arg)))
                .collect::<Result<Vec<_>, _>>()?;
            let keywords = self
                .list(object, "keywords", "Call")?
                .into_iter()
                .map(|keyword| self.recurse(|this| this.keyword(keyword)))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ast::Expr::Call(ast::ExprCall {
                node_index: Default::default(),
                range: Default::default(),
                func: Box::new(self.recurse(|this| this.expr(func))?),
                arguments: ast::Arguments {
                    node_index: Default::default(),
                    range: Default::default(),
                    args: args.into_boxed_slice(),
                    keywords: keywords.into_boxed_slice(),
                    runtime_args: None,
                    runtime_bases: None,
                },
            }))
        } else if self.is_node(object, "Attribute")? {
            let value = self.field(object, "value", "Attribute")?;
            let ctx = self.field(object, "ctx", "Attribute")?;
            Ok(ast::Expr::Attribute(ast::ExprAttribute {
                node_index: Default::default(),
                range: Default::default(),
                value: Box::new(self.recurse(|this| this.expr(value))?),
                attr: ast::Identifier::new(
                    self.string(object, "attr", "Attribute")?,
                    Default::default(),
                ),
                ctx: self.context(ctx)?,
            }))
        } else if self.is_node(object, "List")? || self.is_node(object, "Tuple")? {
            let is_tuple = self.is_node(object, "Tuple")?;
            let elements = self
                .list(object, "elts", if is_tuple { "Tuple" } else { "List" })?
                .into_iter()
                .map(|element| self.recurse(|this| this.expr(element)))
                .collect::<Result<Vec<_>, _>>()?;
            let ctx = self.context(self.field(object, "ctx", "sequence")?)?;
            if is_tuple {
                Ok(ast::Expr::Tuple(ast::ExprTuple {
                    node_index: Default::default(),
                    range: Default::default(),
                    elts: elements,
                    ctx,
                    parenthesized: true,
                    runtime_elts: None,
                }))
            } else {
                Ok(ast::Expr::List(ast::ExprList {
                    node_index: Default::default(),
                    range: Default::default(),
                    elts: elements,
                    ctx,
                    runtime_elts: None,
                }))
            }
        } else if self.is_node(object, "Name")? {
            let ctx = self.context(self.field(object, "ctx", "Name")?)?;
            Ok(ast::Expr::Name(ast::ExprName {
                node_index: Default::default(),
                range: Default::default(),
                id: ast::name::Name::new(self.string(object, "id", "Name")?),
                ctx,
            }))
        } else if self.is_node(object, "Constant")? {
            let value = self.field(object, "value", "Constant")?;
            Ok(ast::Expr::Constant(ast::ExprConstant {
                node_index: Default::default(),
                range: Default::default(),
                value: self.constant_value(value)?,
                kind: None,
                invalid_type: None,
            }))
        } else {
            Err(crate::PyError::type_error(format!(
                "expected some sort of expr, but got {}",
                unsafe { pyre_object::type_name_of(object) }
            )))
        }
    }

    fn keyword(&mut self, object: PyObjectRef) -> AstResult<ast::Keyword> {
        let arg = self
            .optional_field(object, "arg")?
            .map(|value| {
                if !unsafe { pyre_object::is_str(value) } {
                    return Err(crate::PyError::type_error(
                        "AST identifier must be of type str",
                    ));
                }
                Ok(ast::Identifier::new(
                    unsafe { pyre_object::w_str_get_value(value).to_string() },
                    Default::default(),
                ))
            })
            .transpose()?;
        let value = self.field(object, "value", "keyword")?;
        Ok(ast::Keyword {
            node_index: Default::default(),
            range: Default::default(),
            arg,
            value: self.recurse(|this| this.expr(value))?,
        })
    }

    fn constant_value(&self, object: PyObjectRef) -> AstResult<ast::ConstantValue> {
        unsafe {
            if pyre_object::is_none(object) {
                Ok(ast::ConstantValue::None)
            } else if pyre_object::is_bool(object) {
                Ok(ast::ConstantValue::Boolean(pyre_object::w_bool_get_value(
                    object,
                )))
            } else if pyre_object::is_int(object) {
                Ok(ast::ConstantValue::Integer(
                    pyre_object::w_int_get_value(object)
                        .to_string()
                        .into_boxed_str(),
                ))
            } else if pyre_object::is_long(object) {
                Ok(ast::ConstantValue::Integer(
                    pyre_object::w_long_get_value(object)
                        .to_string()
                        .into_boxed_str(),
                ))
            } else if pyre_object::is_float(object) {
                Ok(ast::ConstantValue::Float(pyre_object::w_float_get_value(
                    object,
                )))
            } else if pyre_object::is_str(object) {
                Ok(ast::ConstantValue::Str(
                    pyre_object::w_str_get_value(object)
                        .to_string()
                        .into_boxed_str(),
                ))
            } else if pyre_object::is_bytes(object) {
                Ok(ast::ConstantValue::Bytes(
                    pyre_object::w_bytes_data(object)
                        .to_vec()
                        .into_boxed_slice(),
                ))
            } else if pyre_object::is_ellipsis(object) {
                Ok(ast::ConstantValue::Ellipsis)
            } else {
                Err(crate::PyError::type_error(format!(
                    "got an invalid type in Constant: {}",
                    pyre_object::type_name_of(object)
                )))
            }
        }
    }

    fn context(&self, object: PyObjectRef) -> AstResult<ast::ExprContext> {
        for (name, ctx) in [
            ("Load", ast::ExprContext::Load),
            ("Store", ast::ExprContext::Store),
            ("Del", ast::ExprContext::Del),
            ("Invalid", ast::ExprContext::Invalid),
        ] {
            if self.is_node(object, name)? {
                return Ok(ctx);
            }
        }
        Err(crate::PyError::type_error(
            "expected some sort of expr_context",
        ))
    }

    fn unaryop(&self, object: PyObjectRef) -> AstResult<ast::UnaryOp> {
        for (name, op) in [
            ("Invert", ast::UnaryOp::Invert),
            ("Not", ast::UnaryOp::Not),
            ("UAdd", ast::UnaryOp::UAdd),
            ("USub", ast::UnaryOp::USub),
        ] {
            if self.is_node(object, name)? {
                return Ok(op);
            }
        }
        Err(crate::PyError::type_error("expected some sort of unaryop"))
    }

    fn operator(&self, object: PyObjectRef) -> AstResult<ast::Operator> {
        for (name, op) in [
            ("Add", ast::Operator::Add),
            ("Sub", ast::Operator::Sub),
            ("Mult", ast::Operator::Mult),
            ("MatMult", ast::Operator::MatMult),
            ("Div", ast::Operator::Div),
            ("Mod", ast::Operator::Mod),
            ("Pow", ast::Operator::Pow),
            ("LShift", ast::Operator::LShift),
            ("RShift", ast::Operator::RShift),
            ("BitOr", ast::Operator::BitOr),
            ("BitXor", ast::Operator::BitXor),
            ("BitAnd", ast::Operator::BitAnd),
            ("FloorDiv", ast::Operator::FloorDiv),
        ] {
            if self.is_node(object, name)? {
                return Ok(op);
            }
        }
        Err(crate::PyError::type_error("expected some sort of operator"))
    }
}

pub fn parse_to_object(source: &str, mode: crate::compile::Mode) -> crate::PyResult {
    let parsed = match mode {
        crate::compile::Mode::Eval => parser::parse_expression(source)
            .map(|parsed| ParsedRoot::Expression(parsed.into_syntax())),
        crate::compile::Mode::Exec
        | crate::compile::Mode::Single
        | crate::compile::Mode::BlockExpr => {
            parser::parse_module(source).map(|parsed| ParsedRoot::Module(parsed.into_syntax()))
        }
    }
    .map_err(|error| crate::PyError::syntax_error(error.to_string()))?;

    let ast_module = crate::importing::importhook(
        "_ast",
        PY_NULL,
        PY_NULL,
        0,
        crate::call::take_last_exec_ctx(),
    )?;
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(ast_module);
    let converter = Converter { source, ast_module };
    match parsed {
        ParsedRoot::Expression(module) => converter.node(
            "Expression",
            None,
            &[("body", converter.expr(&module.body)?)],
        ),
        ParsedRoot::Module(module) => {
            let root_name = if matches!(mode, crate::compile::Mode::Single) {
                "Interactive"
            } else {
                "Module"
            };
            let body = converter.stmt_list(&module.body)?;
            if root_name == "Module" {
                converter.node(
                    root_name,
                    None,
                    &[("body", body), ("type_ignores", converter.list(Vec::new()))],
                )
            } else {
                converter.node(root_name, None, &[("body", body)])
            }
        }
    }
}

enum ParsedRoot {
    Module(ast::ModModule),
    Expression(ast::ModExpression),
}

struct Converter<'a> {
    source: &'a str,
    ast_module: PyObjectRef,
}

impl Converter<'_> {
    fn pin(&self, value: PyObjectRef) -> PyObjectRef {
        pyre_object::gc_roots::pin_root(value);
        value
    }

    fn list(&self, values: Vec<PyObjectRef>) -> PyObjectRef {
        self.pin(pyre_object::w_list_new(values))
    }

    fn string(&self, value: &str) -> PyObjectRef {
        self.pin(pyre_object::w_str_new(value))
    }

    fn optional(&self, value: Option<PyObjectRef>) -> PyObjectRef {
        value.unwrap_or_else(pyre_object::w_none)
    }

    fn node(
        &self,
        name: &str,
        range: Option<(u32, u32)>,
        fields: &[(&str, PyObjectRef)],
    ) -> crate::PyResult {
        let node_type = crate::baseobjspace::getattr_str(self.ast_module, name)?;
        let node = self.pin(pyre_object::w_instance_new(node_type));
        for &(field, value) in fields {
            crate::baseobjspace::setattr_str(node, field, value)?;
        }
        if let Some((start, end)) = range {
            let (lineno, col_offset) = self.location(start as usize);
            let (end_lineno, end_col_offset) = self.location(end as usize);
            for (field, value) in [
                ("lineno", lineno),
                ("col_offset", col_offset),
                ("end_lineno", end_lineno),
                ("end_col_offset", end_col_offset),
            ] {
                crate::baseobjspace::setattr_str(
                    node,
                    field,
                    pyre_object::w_int_new(value as i64),
                )?;
            }
        }
        Ok(node)
    }

    fn location(&self, offset: usize) -> (usize, usize) {
        let bytes = self.source.as_bytes();
        let offset = offset.min(bytes.len());
        let prefix = &bytes[..offset];
        let line_start = prefix
            .iter()
            .rposition(|byte| *byte == b'\n')
            .map_or(0, |i| i + 1);
        (
            prefix.iter().filter(|byte| **byte == b'\n').count() + 1,
            offset - line_start,
        )
    }

    fn stmt_list(&self, stmts: &[ast::Stmt]) -> crate::PyResult {
        stmts
            .iter()
            .map(|stmt| self.stmt(stmt))
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn expr_list(&self, exprs: &[ast::Expr]) -> crate::PyResult {
        exprs
            .iter()
            .map(|expr| self.expr(expr))
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn name_list<T: AsRef<str>>(&self, names: &[T]) -> PyObjectRef {
        self.list(
            names
                .iter()
                .map(|name| self.string(name.as_ref()))
                .collect(),
        )
    }

    fn stmt(&self, stmt: &ast::Stmt) -> crate::PyResult {
        use ast::Stmt;
        match stmt {
            Stmt::FunctionDef(node) => {
                let name = if node.is_async {
                    "AsyncFunctionDef"
                } else {
                    "FunctionDef"
                };
                let decorators = node
                    .decorator_list
                    .iter()
                    .map(|d| self.expr(&d.expression))
                    .collect::<Result<Vec<_>, _>>()?;
                self.node(
                    name,
                    Some(range(node.range)),
                    &[
                        ("name", self.string(node.name.as_str())),
                        ("args", self.parameters(&node.parameters)?),
                        ("body", self.stmt_list(&node.body)?),
                        ("decorator_list", self.list(decorators)),
                        (
                            "returns",
                            self.optional(
                                node.returns.as_deref().map(|v| self.expr(v)).transpose()?,
                            ),
                        ),
                        (
                            "type_comment",
                            self.optional(
                                node.runtime_type_comment.as_deref().map(|v| self.string(v)),
                            ),
                        ),
                        (
                            "type_params",
                            self.type_params(node.type_params.as_deref())?,
                        ),
                    ],
                )
            }
            Stmt::ClassDef(node) => {
                let decorators = node
                    .decorator_list
                    .iter()
                    .map(|d| self.expr(&d.expression))
                    .collect::<Result<Vec<_>, _>>()?;
                let (bases, keywords) = if let Some(arguments) = node.arguments.as_deref() {
                    (
                        self.expr_list(&arguments.args)?,
                        self.keyword_list(&arguments.keywords)?,
                    )
                } else {
                    (self.list(Vec::new()), self.list(Vec::new()))
                };
                self.node(
                    "ClassDef",
                    Some(range(node.range)),
                    &[
                        ("name", self.string(node.name.as_str())),
                        ("bases", bases),
                        ("keywords", keywords),
                        ("body", self.stmt_list(&node.body)?),
                        ("decorator_list", self.list(decorators)),
                        (
                            "type_params",
                            self.type_params(node.type_params.as_deref())?,
                        ),
                    ],
                )
            }
            Stmt::Return(node) => self.node(
                "Return",
                Some(range(node.range)),
                &[(
                    "value",
                    self.optional(node.value.as_deref().map(|v| self.expr(v)).transpose()?),
                )],
            ),
            Stmt::Delete(node) => self.node(
                "Delete",
                Some(range(node.range)),
                &[("targets", self.expr_list(&node.targets)?)],
            ),
            Stmt::TypeAlias(node) => self.node(
                "TypeAlias",
                Some(range(node.range)),
                &[
                    ("name", self.expr(&node.name)?),
                    (
                        "type_params",
                        self.type_params(node.type_params.as_deref())?,
                    ),
                    ("value", self.expr(&node.value)?),
                ],
            ),
            Stmt::Assign(node) => self.node(
                "Assign",
                Some(range(node.range)),
                &[
                    ("targets", self.expr_list(&node.targets)?),
                    ("value", self.expr(&node.value)?),
                    (
                        "type_comment",
                        self.optional(node.runtime_type_comment.as_deref().map(|v| self.string(v))),
                    ),
                ],
            ),
            Stmt::AugAssign(node) => self.node(
                "AugAssign",
                Some(range(node.range)),
                &[
                    ("target", self.expr(&node.target)?),
                    ("op", self.operator(node.op)?),
                    ("value", self.expr(&node.value)?),
                ],
            ),
            Stmt::AnnAssign(node) => self.node(
                "AnnAssign",
                Some(range(node.range)),
                &[
                    ("target", self.expr(&node.target)?),
                    ("annotation", self.expr(&node.annotation)?),
                    (
                        "value",
                        self.optional(node.value.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                    (
                        "simple",
                        pyre_object::w_int_new(
                            node.runtime_simple.unwrap_or(node.simple as i32) as i64
                        ),
                    ),
                ],
            ),
            Stmt::For(node) => self.node(
                if node.is_async { "AsyncFor" } else { "For" },
                Some(range(node.range)),
                &[
                    ("target", self.expr(&node.target)?),
                    ("iter", self.expr(&node.iter)?),
                    ("body", self.stmt_list(&node.body)?),
                    ("orelse", self.stmt_list(&node.orelse)?),
                    (
                        "type_comment",
                        self.optional(node.runtime_type_comment.as_deref().map(|v| self.string(v))),
                    ),
                ],
            ),
            Stmt::While(node) => self.node(
                "While",
                Some(range(node.range)),
                &[
                    ("test", self.expr(&node.test)?),
                    ("body", self.stmt_list(&node.body)?),
                    ("orelse", self.stmt_list(&node.orelse)?),
                ],
            ),
            Stmt::If(node) => {
                let mut orelse = Vec::new();
                for clause in node.elif_else_clauses.iter().rev() {
                    let body = self.stmt_list(&clause.body)?;
                    if let Some(test) = clause.test.as_ref() {
                        orelse = vec![self.node(
                            "If",
                            Some(range(clause.range)),
                            &[
                                ("test", self.expr(test)?),
                                ("body", body),
                                ("orelse", self.list(orelse)),
                            ],
                        )?];
                    } else {
                        orelse = unsafe { pyre_object::w_list_items_copy_as_vec(body) };
                    }
                }
                self.node(
                    "If",
                    Some(range(node.range)),
                    &[
                        ("test", self.expr(&node.test)?),
                        ("body", self.stmt_list(&node.body)?),
                        ("orelse", self.list(orelse)),
                    ],
                )
            }
            Stmt::With(node) => self.node(
                if node.is_async { "AsyncWith" } else { "With" },
                Some(range(node.range)),
                &[
                    ("items", self.with_items(&node.items)?),
                    ("body", self.stmt_list(&node.body)?),
                    (
                        "type_comment",
                        self.optional(node.runtime_type_comment.as_deref().map(|v| self.string(v))),
                    ),
                ],
            ),
            Stmt::Raise(node) => self.node(
                "Raise",
                Some(range(node.range)),
                &[
                    (
                        "exc",
                        self.optional(node.exc.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                    (
                        "cause",
                        self.optional(node.cause.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                ],
            ),
            Stmt::Try(node) => self.node(
                if node.is_star { "TryStar" } else { "Try" },
                Some(range(node.range)),
                &[
                    ("body", self.stmt_list(&node.body)?),
                    ("handlers", self.handlers(&node.handlers)?),
                    ("orelse", self.stmt_list(&node.orelse)?),
                    ("finalbody", self.stmt_list(&node.finalbody)?),
                ],
            ),
            Stmt::Assert(node) => self.node(
                "Assert",
                Some(range(node.range)),
                &[
                    ("test", self.expr(&node.test)?),
                    (
                        "msg",
                        self.optional(node.msg.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                ],
            ),
            Stmt::Import(node) => self.node(
                "Import",
                Some(range(node.range)),
                &[("names", self.aliases(&node.names)?)],
            ),
            Stmt::ImportFrom(node) => self.node(
                "ImportFrom",
                Some(range(node.range)),
                &[
                    (
                        "module",
                        self.optional(node.module.as_ref().map(|v| self.string(v.as_str()))),
                    ),
                    ("names", self.aliases(&node.names)?),
                    (
                        "level",
                        pyre_object::w_int_new(
                            node.runtime_level.unwrap_or(node.level as i32) as i64
                        ),
                    ),
                ],
            ),
            Stmt::Global(node) => self.node(
                "Global",
                Some(range(node.range)),
                &[("names", self.name_list(&node.names))],
            ),
            Stmt::Nonlocal(node) => self.node(
                "Nonlocal",
                Some(range(node.range)),
                &[("names", self.name_list(&node.names))],
            ),
            Stmt::Expr(node) => self.node(
                "Expr",
                Some(range(node.range)),
                &[("value", self.expr(&node.value)?)],
            ),
            Stmt::Pass(node) => self.node("Pass", Some(range(node.range)), &[]),
            Stmt::Break(node) => self.node("Break", Some(range(node.range)), &[]),
            Stmt::Continue(node) => self.node("Continue", Some(range(node.range)), &[]),
            Stmt::Match(node) => {
                let cases = node
                    .cases
                    .iter()
                    .map(|case| self.match_case(case))
                    .collect::<Result<Vec<_>, _>>()?;
                self.node(
                    "Match",
                    Some(range(node.range)),
                    &[
                        ("subject", self.expr(&node.subject)?),
                        ("cases", self.list(cases)),
                    ],
                )
            }
            Stmt::IpyEscapeCommand(_) => Err(crate::PyError::not_implemented(
                "AST conversion for IPython escape commands is not implemented",
            )),
        }
    }

    fn expr(&self, expr: &ast::Expr) -> crate::PyResult {
        use ast::Expr;
        match expr {
            Expr::BoolOp(n) => self.node(
                "BoolOp",
                Some(range(n.range)),
                &[
                    ("op", self.boolop(n.op)?),
                    ("values", self.expr_list(&n.values)?),
                ],
            ),
            Expr::Named(n) => self.node(
                "NamedExpr",
                Some(range(n.range)),
                &[
                    ("target", self.expr(&n.target)?),
                    ("value", self.expr(&n.value)?),
                ],
            ),
            Expr::BinOp(n) => self.node(
                "BinOp",
                Some(range(n.range)),
                &[
                    ("left", self.expr(&n.left)?),
                    ("op", self.operator(n.op)?),
                    ("right", self.expr(&n.right)?),
                ],
            ),
            Expr::UnaryOp(n) => self.node(
                "UnaryOp",
                Some(range(n.range)),
                &[
                    ("op", self.unaryop(n.op)?),
                    ("operand", self.expr(&n.operand)?),
                ],
            ),
            Expr::Lambda(n) => self.node(
                "Lambda",
                Some(range(n.range)),
                &[
                    ("args", self.parameters_opt(n.parameters.as_deref())?),
                    ("body", self.expr(&n.body)?),
                ],
            ),
            Expr::If(n) => self.node(
                "IfExp",
                Some(range(n.range)),
                &[
                    ("test", self.expr(&n.test)?),
                    ("body", self.expr(&n.body)?),
                    ("orelse", self.expr(&n.orelse)?),
                ],
            ),
            Expr::Dict(n) => {
                let keys = n
                    .items
                    .iter()
                    .map(|item| {
                        item.key
                            .as_ref()
                            .map(|key| self.expr(key))
                            .transpose()
                            .map(|v| self.optional(v))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let values = n
                    .items
                    .iter()
                    .map(|item| self.expr(&item.value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.node(
                    "Dict",
                    Some(range(n.range)),
                    &[("keys", self.list(keys)), ("values", self.list(values))],
                )
            }
            Expr::Set(n) => self.node(
                "Set",
                Some(range(n.range)),
                &[("elts", self.expr_list(&n.elts)?)],
            ),
            Expr::ListComp(n) => self.node(
                "ListComp",
                Some(range(n.range)),
                &[
                    ("elt", self.expr(&n.elt)?),
                    ("generators", self.comprehensions(&n.generators)?),
                ],
            ),
            Expr::SetComp(n) => self.node(
                "SetComp",
                Some(range(n.range)),
                &[
                    ("elt", self.expr(&n.elt)?),
                    ("generators", self.comprehensions(&n.generators)?),
                ],
            ),
            Expr::DictComp(n) => self.node(
                "DictComp",
                Some(range(n.range)),
                &[
                    ("key", self.expr(&n.key)?),
                    ("value", self.expr(&n.value)?),
                    ("generators", self.comprehensions(&n.generators)?),
                ],
            ),
            Expr::Generator(n) => self.node(
                "GeneratorExp",
                Some(range(n.range)),
                &[
                    ("elt", self.expr(&n.elt)?),
                    ("generators", self.comprehensions(&n.generators)?),
                ],
            ),
            Expr::Await(n) => self.node(
                "Await",
                Some(range(n.range)),
                &[("value", self.expr(&n.value)?)],
            ),
            Expr::Yield(n) => self.node(
                "Yield",
                Some(range(n.range)),
                &[(
                    "value",
                    self.optional(n.value.as_deref().map(|v| self.expr(v)).transpose()?),
                )],
            ),
            Expr::YieldFrom(n) => self.node(
                "YieldFrom",
                Some(range(n.range)),
                &[("value", self.expr(&n.value)?)],
            ),
            Expr::Compare(n) => {
                let ops = n
                    .ops
                    .iter()
                    .map(|op| self.cmpop(*op))
                    .collect::<Result<Vec<_>, _>>()?;
                self.node(
                    "Compare",
                    Some(range(n.range)),
                    &[
                        ("left", self.expr(&n.left)?),
                        ("ops", self.list(ops)),
                        ("comparators", self.expr_list(&n.comparators)?),
                    ],
                )
            }
            Expr::Call(n) => self.node(
                "Call",
                Some(range(n.range)),
                &[
                    ("func", self.expr(&n.func)?),
                    ("args", self.expr_list(&n.arguments.args)?),
                    ("keywords", self.keyword_list(&n.arguments.keywords)?),
                ],
            ),
            Expr::StringLiteral(n) => self.constant(
                range(n.range),
                self.string(n.value.to_str()),
                if n.value.is_unicode() {
                    self.string("u")
                } else {
                    pyre_object::w_none()
                },
            ),
            Expr::BytesLiteral(n) => self.constant(
                range(n.range),
                self.pin(pyre_object::w_bytes_from_bytes(
                    &n.value.bytes().collect::<Vec<_>>(),
                )),
                pyre_object::w_none(),
            ),
            Expr::NumberLiteral(n) => self.constant(
                range(n.range),
                self.number(&n.value)?,
                pyre_object::w_none(),
            ),
            Expr::BooleanLiteral(n) => self.constant(
                range(n.range),
                pyre_object::w_bool_from(n.value),
                pyre_object::w_none(),
            ),
            Expr::NoneLiteral(n) => {
                self.constant(range(n.range), pyre_object::w_none(), pyre_object::w_none())
            }
            Expr::EllipsisLiteral(n) => self.constant(
                range(n.range),
                pyre_object::w_ellipsis(),
                pyre_object::w_none(),
            ),
            Expr::Constant(n) => self.constant(
                range(n.range),
                self.constant_value(&n.value)?,
                self.optional(n.kind.as_deref().map(|v| self.string(v))),
            ),
            Expr::Attribute(n) => self.node(
                "Attribute",
                Some(range(n.range)),
                &[
                    ("value", self.expr(&n.value)?),
                    ("attr", self.string(n.attr.as_str())),
                    ("ctx", self.context(n.ctx)?),
                ],
            ),
            Expr::Subscript(n) => self.node(
                "Subscript",
                Some(range(n.range)),
                &[
                    ("value", self.expr(&n.value)?),
                    ("slice", self.expr(&n.slice)?),
                    ("ctx", self.context(n.ctx)?),
                ],
            ),
            Expr::Starred(n) => self.node(
                "Starred",
                Some(range(n.range)),
                &[
                    ("value", self.expr(&n.value)?),
                    ("ctx", self.context(n.ctx)?),
                ],
            ),
            Expr::Name(n) => self.node(
                "Name",
                Some(range(n.range)),
                &[
                    ("id", self.string(n.id.as_str())),
                    ("ctx", self.context(n.ctx)?),
                ],
            ),
            Expr::List(n) => self.node(
                "List",
                Some(range(n.range)),
                &[
                    ("elts", self.expr_list(&n.elts)?),
                    ("ctx", self.context(n.ctx)?),
                ],
            ),
            Expr::Tuple(n) => self.node(
                "Tuple",
                Some(range(n.range)),
                &[
                    ("elts", self.expr_list(&n.elts)?),
                    ("ctx", self.context(n.ctx)?),
                ],
            ),
            Expr::Slice(n) => self.node(
                "Slice",
                Some(range(n.range)),
                &[
                    (
                        "lower",
                        self.optional(n.lower.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                    (
                        "upper",
                        self.optional(n.upper.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                    (
                        "step",
                        self.optional(n.step.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                ],
            ),
            Expr::FString(n) => self.fstring(n),
            Expr::TString(_) | Expr::IpyEscapeCommand(_) => Err(crate::PyError::not_implemented(
                "AST conversion for template strings is not implemented",
            )),
        }
    }

    fn fstring(&self, node: &ast::ExprFString) -> crate::PyResult {
        if let Some(values) = node.runtime_joined_str.as_deref() {
            return self.node(
                "JoinedStr",
                Some(range(node.range)),
                &[("values", self.expr_list(values)?)],
            );
        }
        if let Some(values) = node.runtime_values.as_deref() {
            let values = values
                .iter()
                .map(|value| {
                    value
                        .as_ref()
                        .map(|value| self.expr(value))
                        .transpose()
                        .map(|value| self.optional(value))
                })
                .collect::<Result<Vec<_>, _>>()?;
            return self.node(
                "JoinedStr",
                Some(range(node.range)),
                &[("values", self.list(values))],
            );
        }

        let mut values = Vec::new();
        for part in node.value.iter() {
            match part {
                ast::FStringPart::Literal(literal) => values.push(self.constant(
                    range(literal.range),
                    self.string(&literal.value),
                    pyre_object::w_none(),
                )?),
                ast::FStringPart::FString(fstring) => {
                    values.extend(self.interpolated_elements(&fstring.elements)?);
                }
            }
        }
        self.node(
            "JoinedStr",
            Some(range(node.range)),
            &[("values", self.list(values))],
        )
    }

    fn interpolated_elements(
        &self,
        elements: &[ast::InterpolatedStringElement],
    ) -> Result<Vec<PyObjectRef>, crate::PyError> {
        let mut values = Vec::new();
        for element in elements {
            match element {
                ast::InterpolatedStringElement::Literal(literal) => values.push(self.constant(
                    range(literal.range),
                    self.string(&literal.value),
                    pyre_object::w_none(),
                )?),
                ast::InterpolatedStringElement::Interpolation(interpolation) => {
                    let format_spec = interpolation
                        .format_spec
                        .as_deref()
                        .map(|spec| {
                            let values = self.interpolated_elements(&spec.elements)?;
                            self.node(
                                "JoinedStr",
                                Some(range(spec.range)),
                                &[("values", self.list(values))],
                            )
                        })
                        .transpose()?;
                    values.push(self.node(
                        "FormattedValue",
                        Some(range(interpolation.range)),
                        &[
                            ("value", self.expr(&interpolation.expression)?),
                            (
                                "conversion",
                                pyre_object::w_int_new(interpolation.conversion as i8 as i64),
                            ),
                            ("format_spec", self.optional(format_spec)),
                        ],
                    )?);
                }
            }
        }
        Ok(values)
    }

    fn match_case(&self, case: &ast::MatchCase) -> crate::PyResult {
        self.node(
            "match_case",
            None,
            &[
                ("pattern", self.pattern(&case.pattern)?),
                (
                    "guard",
                    self.optional(case.guard.as_deref().map(|v| self.expr(v)).transpose()?),
                ),
                ("body", self.stmt_list(&case.body)?),
            ],
        )
    }

    fn pattern(&self, pattern: &ast::Pattern) -> crate::PyResult {
        match pattern {
            ast::Pattern::MatchValue(node) => self.node(
                "MatchValue",
                Some(range(node.range)),
                &[("value", self.expr(&node.value)?)],
            ),
            ast::Pattern::MatchSingleton(node) => self.node(
                "MatchSingleton",
                Some(range(node.range)),
                &[(
                    "value",
                    match node.value {
                        ast::Singleton::None => pyre_object::w_none(),
                        ast::Singleton::True => pyre_object::w_bool_from(true),
                        ast::Singleton::False => pyre_object::w_bool_from(false),
                    },
                )],
            ),
            ast::Pattern::MatchSequence(node) => self.node(
                "MatchSequence",
                Some(range(node.range)),
                &[("patterns", self.pattern_list(&node.patterns)?)],
            ),
            ast::Pattern::MatchMapping(node) => self.node(
                "MatchMapping",
                Some(range(node.range)),
                &[
                    ("keys", self.expr_list(&node.keys)?),
                    ("patterns", self.pattern_list(&node.patterns)?),
                    (
                        "rest",
                        self.optional(node.rest.as_ref().map(|name| self.string(name.as_str()))),
                    ),
                ],
            ),
            ast::Pattern::MatchClass(node) => {
                let kwd_attrs = node
                    .arguments
                    .keywords
                    .iter()
                    .map(|keyword| self.string(keyword.attr.as_str()))
                    .collect();
                let kwd_patterns = node
                    .arguments
                    .keywords
                    .iter()
                    .map(|keyword| self.pattern(&keyword.pattern))
                    .collect::<Result<Vec<_>, _>>()?;
                self.node(
                    "MatchClass",
                    Some(range(node.range)),
                    &[
                        ("cls", self.expr(&node.cls)?),
                        ("patterns", self.pattern_list(&node.arguments.patterns)?),
                        ("kwd_attrs", self.list(kwd_attrs)),
                        ("kwd_patterns", self.list(kwd_patterns)),
                    ],
                )
            }
            ast::Pattern::MatchStar(node) => self.node(
                "MatchStar",
                Some(range(node.range)),
                &[(
                    "name",
                    self.optional(node.name.as_ref().map(|name| self.string(name.as_str()))),
                )],
            ),
            ast::Pattern::MatchAs(node) => self.node(
                "MatchAs",
                Some(range(node.range)),
                &[
                    (
                        "pattern",
                        self.optional(
                            node.pattern
                                .as_deref()
                                .map(|pattern| self.pattern(pattern))
                                .transpose()?,
                        ),
                    ),
                    (
                        "name",
                        self.optional(node.name.as_ref().map(|name| self.string(name.as_str()))),
                    ),
                ],
            ),
            ast::Pattern::MatchOr(node) => self.node(
                "MatchOr",
                Some(range(node.range)),
                &[("patterns", self.pattern_list(&node.patterns)?)],
            ),
        }
    }

    fn pattern_list(&self, patterns: &[ast::Pattern]) -> crate::PyResult {
        patterns
            .iter()
            .map(|pattern| self.pattern(pattern))
            .collect::<Result<Vec<_>, _>>()
            .map(|patterns| self.list(patterns))
    }

    fn constant(
        &self,
        range: (u32, u32),
        value: PyObjectRef,
        kind: PyObjectRef,
    ) -> crate::PyResult {
        self.node("Constant", Some(range), &[("value", value), ("kind", kind)])
    }

    fn number(&self, value: &ast::Number) -> crate::PyResult {
        Ok(match value {
            ast::Number::Int(value) => {
                let int_type =
                    crate::typedef::gettypefor(&pyre_object::INT_TYPE).unwrap_or(PY_NULL);
                crate::call::call_function_impl_result(
                    int_type,
                    &[self.string(&value.to_string())],
                )?
            }
            ast::Number::Float(value) => self.pin(pyre_object::w_float_new(*value)),
            ast::Number::Complex { real, imag } => {
                self.pin(pyre_object::w_complex_new(*real, *imag))
            }
        })
    }

    fn constant_value(&self, value: &ast::ConstantValue) -> crate::PyResult {
        Ok(match value {
            ast::ConstantValue::None => pyre_object::w_none(),
            ast::ConstantValue::Boolean(value) => pyre_object::w_bool_from(*value),
            ast::ConstantValue::Str(value) => self.string(value),
            ast::ConstantValue::Bytes(value) => self.pin(pyre_object::w_bytes_from_bytes(value)),
            ast::ConstantValue::Integer(value) => {
                let int_type =
                    crate::typedef::gettypefor(&pyre_object::INT_TYPE).unwrap_or(PY_NULL);
                crate::call::call_function_impl_result(int_type, &[self.string(value)])?
            }
            ast::ConstantValue::Float(value) => self.pin(pyre_object::w_float_new(*value)),
            ast::ConstantValue::Complex { real, imag } => {
                self.pin(pyre_object::w_complex_new(*real, *imag))
            }
            ast::ConstantValue::Ellipsis => pyre_object::w_ellipsis(),
            ast::ConstantValue::Tuple(values) => {
                let values = values
                    .iter()
                    .map(|v| self.constant_value(v))
                    .collect::<Result<Vec<_>, _>>()?;
                self.pin(pyre_object::w_tuple_new(values))
            }
            ast::ConstantValue::Frozenset(_) => {
                return Err(crate::PyError::not_implemented(
                    "frozenset AST constants are not implemented",
                ));
            }
        })
    }

    fn singleton(&self, name: &str) -> crate::PyResult {
        let typ = crate::baseobjspace::getattr_str(self.ast_module, name)?;
        Ok(self.pin(pyre_object::w_instance_new(typ)))
    }

    fn context(&self, value: ast::ExprContext) -> crate::PyResult {
        self.singleton(match value {
            ast::ExprContext::Load => "Load",
            ast::ExprContext::Store => "Store",
            ast::ExprContext::Del => "Del",
            ast::ExprContext::Invalid => "Load",
        })
    }
    fn boolop(&self, value: ast::BoolOp) -> crate::PyResult {
        self.singleton(match value {
            ast::BoolOp::And => "And",
            ast::BoolOp::Or => "Or",
        })
    }
    fn operator(&self, value: ast::Operator) -> crate::PyResult {
        self.singleton(match value {
            ast::Operator::Add => "Add",
            ast::Operator::Sub => "Sub",
            ast::Operator::Mult => "Mult",
            ast::Operator::MatMult => "MatMult",
            ast::Operator::Div => "Div",
            ast::Operator::Mod => "Mod",
            ast::Operator::Pow => "Pow",
            ast::Operator::LShift => "LShift",
            ast::Operator::RShift => "RShift",
            ast::Operator::BitOr => "BitOr",
            ast::Operator::BitXor => "BitXor",
            ast::Operator::BitAnd => "BitAnd",
            ast::Operator::FloorDiv => "FloorDiv",
        })
    }
    fn unaryop(&self, value: ast::UnaryOp) -> crate::PyResult {
        self.singleton(match value {
            ast::UnaryOp::Invert => "Invert",
            ast::UnaryOp::Not => "Not",
            ast::UnaryOp::UAdd => "UAdd",
            ast::UnaryOp::USub => "USub",
        })
    }
    fn cmpop(&self, value: ast::CmpOp) -> crate::PyResult {
        self.singleton(match value {
            ast::CmpOp::Eq => "Eq",
            ast::CmpOp::NotEq => "NotEq",
            ast::CmpOp::Lt => "Lt",
            ast::CmpOp::LtE => "LtE",
            ast::CmpOp::Gt => "Gt",
            ast::CmpOp::GtE => "GtE",
            ast::CmpOp::Is => "Is",
            ast::CmpOp::IsNot => "IsNot",
            ast::CmpOp::In => "In",
            ast::CmpOp::NotIn => "NotIn",
        })
    }

    fn parameters_opt(&self, parameters: Option<&ast::Parameters>) -> crate::PyResult {
        match parameters {
            Some(p) => self.parameters(p),
            None => self.parameters(&ast::Parameters::default()),
        }
    }

    fn parameters(&self, p: &ast::Parameters) -> crate::PyResult {
        let posonlyargs = p
            .posonlyargs
            .iter()
            .map(|p| self.parameter(&p.parameter))
            .collect::<Result<Vec<_>, _>>()?;
        let args = p
            .args
            .iter()
            .map(|p| self.parameter(&p.parameter))
            .collect::<Result<Vec<_>, _>>()?;
        let kwonlyargs = p
            .kwonlyargs
            .iter()
            .map(|p| self.parameter(&p.parameter))
            .collect::<Result<Vec<_>, _>>()?;
        let mut defaults = Vec::new();
        defaults.extend(
            p.posonlyargs
                .iter()
                .chain(&p.args)
                .filter_map(|p| p.default.as_deref())
                .map(|v| self.expr(v))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let kw_defaults = p
            .kwonlyargs
            .iter()
            .map(|p| {
                p.default
                    .as_deref()
                    .map(|v| self.expr(v))
                    .transpose()
                    .map(|v| self.optional(v))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.node(
            "arguments",
            None,
            &[
                ("posonlyargs", self.list(posonlyargs)),
                ("args", self.list(args)),
                (
                    "vararg",
                    self.optional(p.vararg.as_deref().map(|v| self.parameter(v)).transpose()?),
                ),
                ("kwonlyargs", self.list(kwonlyargs)),
                ("kw_defaults", self.list(kw_defaults)),
                (
                    "kwarg",
                    self.optional(p.kwarg.as_deref().map(|v| self.parameter(v)).transpose()?),
                ),
                ("defaults", self.list(defaults)),
            ],
        )
    }

    fn parameter(&self, p: &ast::Parameter) -> crate::PyResult {
        self.node(
            "arg",
            Some(range(p.range)),
            &[
                ("arg", self.string(p.name.as_str())),
                (
                    "annotation",
                    self.optional(p.annotation.as_deref().map(|v| self.expr(v)).transpose()?),
                ),
                ("type_comment", pyre_object::w_none()),
            ],
        )
    }

    fn keyword_list(&self, keywords: &[ast::Keyword]) -> crate::PyResult {
        keywords
            .iter()
            .map(|k| {
                self.node(
                    "keyword",
                    Some(range(k.range)),
                    &[
                        (
                            "arg",
                            self.optional(k.arg.as_ref().map(|v| self.string(v.as_str()))),
                        ),
                        ("value", self.expr(&k.value)?),
                    ],
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn aliases(&self, aliases: &[ast::Alias]) -> crate::PyResult {
        aliases
            .iter()
            .map(|a| {
                self.node(
                    "alias",
                    Some(range(a.range)),
                    &[
                        ("name", self.string(a.name.as_str())),
                        (
                            "asname",
                            self.optional(a.asname.as_ref().map(|v| self.string(v.as_str()))),
                        ),
                    ],
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn with_items(&self, items: &[ast::WithItem]) -> crate::PyResult {
        items
            .iter()
            .map(|item| {
                self.node(
                    "withitem",
                    None,
                    &[
                        ("context_expr", self.expr(&item.context_expr)?),
                        (
                            "optional_vars",
                            self.optional(
                                item.optional_vars
                                    .as_deref()
                                    .map(|v| self.expr(v))
                                    .transpose()?,
                            ),
                        ),
                    ],
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn comprehensions(&self, comprehensions: &[ast::Comprehension]) -> crate::PyResult {
        comprehensions
            .iter()
            .map(|c| {
                self.node(
                    "comprehension",
                    None,
                    &[
                        ("target", self.expr(&c.target)?),
                        ("iter", self.expr(&c.iter)?),
                        ("ifs", self.expr_list(&c.ifs)?),
                        ("is_async", pyre_object::w_int_new(c.is_async as i64)),
                    ],
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn handlers(&self, handlers: &[ast::ExceptHandler]) -> crate::PyResult {
        handlers
            .iter()
            .map(|handler| match handler {
                ast::ExceptHandler::ExceptHandler(h) => self.node(
                    "ExceptHandler",
                    Some(range(h.range)),
                    &[
                        (
                            "type",
                            self.optional(h.type_.as_deref().map(|v| self.expr(v)).transpose()?),
                        ),
                        (
                            "name",
                            self.optional(h.name.as_ref().map(|v| self.string(v.as_str()))),
                        ),
                        ("body", self.stmt_list(&h.body)?),
                    ],
                ),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn type_params(&self, params: Option<&ast::TypeParams>) -> crate::PyResult {
        let Some(params) = params else {
            return Ok(self.list(Vec::new()));
        };
        params
            .type_params
            .iter()
            .map(|param| match param {
                ast::TypeParam::TypeVar(p) => self.node(
                    "TypeVar",
                    Some(range(p.range)),
                    &[
                        ("name", self.string(p.name.as_str())),
                        (
                            "bound",
                            self.optional(p.bound.as_deref().map(|v| self.expr(v)).transpose()?),
                        ),
                        (
                            "default_value",
                            self.optional(p.default.as_deref().map(|v| self.expr(v)).transpose()?),
                        ),
                    ],
                ),
                ast::TypeParam::TypeVarTuple(p) => self.node(
                    "TypeVarTuple",
                    Some(range(p.range)),
                    &[
                        ("name", self.string(p.name.as_str())),
                        (
                            "default_value",
                            self.optional(p.default.as_deref().map(|v| self.expr(v)).transpose()?),
                        ),
                    ],
                ),
                ast::TypeParam::ParamSpec(p) => self.node(
                    "ParamSpec",
                    Some(range(p.range)),
                    &[
                        ("name", self.string(p.name.as_str())),
                        (
                            "default_value",
                            self.optional(p.default.as_deref().map(|v| self.expr(v)).transpose()?),
                        ),
                    ],
                ),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }
}

fn range(range: impl RangeParts) -> (u32, u32) {
    range.parts()
}

trait RangeParts {
    fn parts(self) -> (u32, u32);
}

impl RangeParts for ruff_text_size::TextRange {
    fn parts(self) -> (u32, u32) {
        (self.start().to_u32(), self.end().to_u32())
    }
}
