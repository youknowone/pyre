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
    // compiling.py:73 — the tree is walked before it reaches the compiler.
    crate::astcompiler::validate::validate_ast(&module)?;
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
                "{node} field {field:?} must be a list, not a {}",
                class_name(value)
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
        let node = if self.is_node(object, "Module")? {
            "Module"
        } else if self.is_node(object, "Interactive")? {
            "Interactive"
        } else if self.is_node(object, "Expression")? {
            let body = self.field(object, "body", "Expression")?;
            return Ok(ast::Mod::Expression(ast::ModExpression {
                node_index: Default::default(),
                range: Default::default(),
                body: Box::new(self.recurse(|this| this.expr(body))?),
            }));
        } else {
            return Err(crate::PyError::type_error(format!(
                "expected some sort of mod, but got {}",
                unsafe { pyre_object::type_name_of(object) }
            )));
        };
        Ok(ast::Mod::Module(ast::ModModule {
            node_index: Default::default(),
            range: Default::default(),
            body: self.body(object, "body", node)?,
            runtime_body: None,
        }))
    }

    fn stmt(&mut self, object: PyObjectRef) -> AstResult<ast::Stmt> {
        if self.is_node(object, "FunctionDef")? || self.is_node(object, "AsyncFunctionDef")? {
            let is_async = self.is_node(object, "AsyncFunctionDef")?;
            let node = if is_async {
                "AsyncFunctionDef"
            } else {
                "FunctionDef"
            };
            let name = self.identifier(object, "name", node)?;
            let args = self.field(object, "args", node)?;
            let parameters = Box::new(self.recurse(|this| this.parameters(args))?);
            let body = self.body(object, "body", node)?;
            let decorator_list = self.decorators(object, node)?;
            let returns = self.opt_expr(object, "returns")?;
            let type_params = self.type_params(object, node)?;
            Ok(ast::Stmt::FunctionDef(ast::StmtFunctionDef {
                node_index: Default::default(),
                range: Default::default(),
                is_async,
                decorator_list,
                name,
                type_params,
                parameters,
                returns,
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
                .map(|value| {
                    self.require_node(value, "expression")?;
                    self.recurse(|this| this.expr(value))
                })
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
        } else if self.is_node(object, "ClassDef")? {
            let name = self.identifier(object, "name", "ClassDef")?;
            let bases = self.exprs(object, "bases", "ClassDef")?;
            let keywords = self
                .list(object, "keywords", "ClassDef")?
                .into_iter()
                .map(|keyword| self.recurse(|this| this.keyword(keyword)))
                .collect::<Result<Vec<_>, _>>()?;
            let body = self.body(object, "body", "ClassDef")?;
            let decorator_list = self.decorators(object, "ClassDef")?;
            let type_params = self.type_params(object, "ClassDef")?;
            // An absent argument list and an empty one are different trees, and
            // only the former elides the parentheses.
            let arguments = if bases.is_empty() && keywords.is_empty() {
                None
            } else {
                Some(Box::new(ast::Arguments {
                    node_index: Default::default(),
                    range: Default::default(),
                    args: bases.into_boxed_slice(),
                    keywords: keywords.into_boxed_slice(),
                    runtime_args: None,
                    runtime_bases: None,
                }))
            };
            Ok(ast::Stmt::ClassDef(ast::StmtClassDef {
                node_index: Default::default(),
                range: Default::default(),
                decorator_list,
                name,
                type_params,
                arguments,
                body,
                runtime_decorator_list: None,
                runtime_body: None,
            }))
        } else if self.is_node(object, "Delete")? {
            Ok(ast::Stmt::Delete(ast::StmtDelete {
                node_index: Default::default(),
                range: Default::default(),
                targets: self.exprs(object, "targets", "Delete")?,
                runtime_targets: None,
            }))
        } else if self.is_node(object, "TypeAlias")? {
            let name = self.req_expr(object, "name", "TypeAlias")?;
            let type_params = self.type_params(object, "TypeAlias")?;
            let value = self.req_expr(object, "value", "TypeAlias")?;
            Ok(ast::Stmt::TypeAlias(ast::StmtTypeAlias {
                node_index: Default::default(),
                range: Default::default(),
                name,
                type_params,
                value,
            }))
        } else if self.is_node(object, "AugAssign")? {
            let target = self.req_expr(object, "target", "AugAssign")?;
            let op = self.field(object, "op", "AugAssign")?;
            let value = self.req_expr(object, "value", "AugAssign")?;
            Ok(ast::Stmt::AugAssign(ast::StmtAugAssign {
                node_index: Default::default(),
                range: Default::default(),
                target,
                op: self.operator(op)?,
                value,
            }))
        } else if self.is_node(object, "AnnAssign")? {
            let target = self.req_expr(object, "target", "AnnAssign")?;
            let annotation = self.req_expr(object, "annotation", "AnnAssign")?;
            let value = self.opt_expr(object, "value")?;
            let simple = self.int_field(object, "simple", "AnnAssign")?;
            Ok(ast::Stmt::AnnAssign(ast::StmtAnnAssign {
                node_index: Default::default(),
                range: Default::default(),
                target,
                annotation,
                value,
                simple: simple != 0,
                runtime_simple: None,
            }))
        } else if self.is_node(object, "For")? || self.is_node(object, "AsyncFor")? {
            let is_async = self.is_node(object, "AsyncFor")?;
            let node = if is_async { "AsyncFor" } else { "For" };
            let target = self.req_expr(object, "target", node)?;
            let iter = self.req_expr(object, "iter", node)?;
            let body = self.body(object, "body", node)?;
            let orelse = self.body(object, "orelse", node)?;
            Ok(ast::Stmt::For(ast::StmtFor {
                node_index: Default::default(),
                range: Default::default(),
                is_async,
                target,
                iter,
                body,
                orelse,
                runtime_type_comment: None,
                runtime_type_comment_bytes: None,
                runtime_body: None,
                runtime_orelse: None,
            }))
        } else if self.is_node(object, "While")? {
            let test = self.req_expr(object, "test", "While")?;
            let body = self.body(object, "body", "While")?;
            let orelse = self.body(object, "orelse", "While")?;
            Ok(ast::Stmt::While(ast::StmtWhile {
                node_index: Default::default(),
                range: Default::default(),
                test,
                body,
                orelse,
                runtime_body: None,
                runtime_orelse: None,
            }))
        } else if self.is_node(object, "If")? {
            let test = self.req_expr(object, "test", "If")?;
            let body = self.body(object, "body", "If")?;
            // `If` carries its alternatives as a nested `orelse`, while the
            // compiler AST keeps them in one flat clause list.  An `elif` and an
            // `else` holding a single `if` are indistinguishable here, exactly as
            // they are to the parser, so both flatten the same way.
            let mut elif_else_clauses = Vec::new();
            let mut orelse = self.list(object, "orelse", "If")?;
            while !orelse.is_empty() {
                if orelse.len() == 1 && self.is_node(orelse[0], "If")? {
                    let nested = orelse[0];
                    let clause_test = self.req_expr(nested, "test", "If")?;
                    let clause_body = self.body(nested, "body", "If")?;
                    elif_else_clauses.push(ast::ElifElseClause {
                        range: Default::default(),
                        node_index: Default::default(),
                        test: Some(*clause_test),
                        body: clause_body,
                        runtime_body: None,
                        runtime_orelse: None,
                    });
                    orelse = self.list(nested, "orelse", "If")?;
                } else {
                    let values = std::mem::take(&mut orelse);
                    let clause_body = values
                        .into_iter()
                        .map(|value| {
                            self.require_node(value, "statement")?;
                            self.recurse(|this| this.stmt(value))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    elif_else_clauses.push(ast::ElifElseClause {
                        range: Default::default(),
                        node_index: Default::default(),
                        test: None,
                        body: clause_body,
                        runtime_body: None,
                        runtime_orelse: None,
                    });
                }
            }
            Ok(ast::Stmt::If(ast::StmtIf {
                node_index: Default::default(),
                range: Default::default(),
                test,
                body,
                elif_else_clauses,
                runtime_body: None,
            }))
        } else if self.is_node(object, "With")? || self.is_node(object, "AsyncWith")? {
            let is_async = self.is_node(object, "AsyncWith")?;
            let node = if is_async { "AsyncWith" } else { "With" };
            let items = self
                .list(object, "items", node)?
                .into_iter()
                .map(|item| self.recurse(|this| this.with_item(item)))
                .collect::<Result<Vec<_>, _>>()?;
            let body = self.body(object, "body", node)?;
            Ok(ast::Stmt::With(ast::StmtWith {
                node_index: Default::default(),
                range: Default::default(),
                is_async,
                items,
                body,
                runtime_type_comment: None,
                runtime_type_comment_bytes: None,
                runtime_body: None,
            }))
        } else if self.is_node(object, "Raise")? {
            Ok(ast::Stmt::Raise(ast::StmtRaise {
                node_index: Default::default(),
                range: Default::default(),
                exc: self.opt_expr(object, "exc")?,
                cause: self.opt_expr(object, "cause")?,
            }))
        } else if self.is_node(object, "Try")? || self.is_node(object, "TryStar")? {
            let is_star = self.is_node(object, "TryStar")?;
            let node = if is_star { "TryStar" } else { "Try" };
            let body = self.body(object, "body", node)?;
            let handlers = self
                .list(object, "handlers", node)?
                .into_iter()
                .map(|handler| self.recurse(|this| this.handler(handler)))
                .collect::<Result<Vec<_>, _>>()?;
            let orelse = self.body(object, "orelse", node)?;
            let finalbody = self.body(object, "finalbody", node)?;
            Ok(ast::Stmt::Try(ast::StmtTry {
                node_index: Default::default(),
                range: Default::default(),
                body,
                handlers,
                orelse,
                finalbody,
                is_star,
                runtime_body: None,
                runtime_handlers: None,
                runtime_orelse: None,
                runtime_finalbody: None,
            }))
        } else if self.is_node(object, "Assert")? {
            let test = self.req_expr(object, "test", "Assert")?;
            let msg = self.opt_expr(object, "msg")?;
            Ok(ast::Stmt::Assert(ast::StmtAssert {
                node_index: Default::default(),
                range: Default::default(),
                test,
                msg,
            }))
        } else if self.is_node(object, "Import")? {
            Ok(ast::Stmt::Import(ast::StmtImport {
                node_index: Default::default(),
                range: Default::default(),
                names: self.aliases(object, "Import")?,
                is_lazy: false,
            }))
        } else if self.is_node(object, "ImportFrom")? {
            let module = self.opt_identifier(object, "module")?;
            let names = self.aliases(object, "ImportFrom")?;
            // `level` is optional on a hand-built node and defaults to absolute.
            let level = match self.optional_field(object, "level")? {
                Some(value) => crate::builtins::space_index_w(value)?,
                None => 0,
            };
            if level < 0 {
                return Err(crate::PyError::value_error("Negative ImportFrom level"));
            }
            // The field is read as a Python index, so a value past the range a
            // level is stored in has to stop here rather than wrap and turn a
            // relative import into an absolute one.
            let level = u32::try_from(level).map_err(|_| {
                crate::PyError::overflow_error("Python int too large to convert to C int")
            })?;
            Ok(ast::Stmt::ImportFrom(ast::StmtImportFrom {
                node_index: Default::default(),
                range: Default::default(),
                module,
                names,
                level,
                is_lazy: false,
                runtime_level: None,
            }))
        } else if self.is_node(object, "Global")? {
            Ok(ast::Stmt::Global(ast::StmtGlobal {
                node_index: Default::default(),
                range: Default::default(),
                names: self.identifiers(object, "names", "Global")?,
            }))
        } else if self.is_node(object, "Nonlocal")? {
            Ok(ast::Stmt::Nonlocal(ast::StmtNonlocal {
                node_index: Default::default(),
                range: Default::default(),
                names: self.identifiers(object, "names", "Nonlocal")?,
            }))
        } else if self.is_node(object, "Break")? {
            Ok(ast::Stmt::Break(ast::StmtBreak {
                node_index: Default::default(),
                range: Default::default(),
            }))
        } else if self.is_node(object, "Continue")? {
            Ok(ast::Stmt::Continue(ast::StmtContinue {
                node_index: Default::default(),
                range: Default::default(),
            }))
        } else if self.is_node(object, "Match")? {
            let subject = self.req_expr(object, "subject", "Match")?;
            let cases = self
                .list(object, "cases", "Match")?
                .into_iter()
                .map(|case| self.recurse(|this| this.match_case(case)))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ast::Stmt::Match(ast::StmtMatch {
                node_index: Default::default(),
                range: Default::default(),
                subject,
                cases,
            }))
        } else {
            Err(crate::PyError::type_error(format!(
                "expected some sort of stmt, but got {}",
                unsafe { pyre_object::type_name_of(object) }
            )))
        }
    }

    fn parameters(&mut self, object: PyObjectRef) -> AstResult<ast::Parameters> {
        let posonlyargs = self.parameter_list(object, "posonlyargs")?;
        let args = self.parameter_list(object, "args")?;
        let kwonlyargs = self.parameter_list(object, "kwonlyargs")?;
        let vararg = self.opt_parameter(object, "vararg")?;
        let kwarg = self.opt_parameter(object, "kwarg")?;
        // `defaults` covers the tail of posonlyargs ++ args, while `kw_defaults`
        // runs alongside kwonlyargs with a hole for every parameter that has
        // none.  The compiler AST carries each default on its own parameter, so
        // both lists are distributed here.
        let defaults = self.exprs(object, "defaults", "arguments")?;
        let kw_defaults = self.list(object, "kw_defaults", "arguments")?;
        if kw_defaults.len() != kwonlyargs.len() {
            return Err(crate::PyError::value_error(
                "length of kwonlyargs is not the same as kw_defaults on arguments",
            ));
        }
        let mut positional: Vec<ast::Parameter> = posonlyargs;
        let posonly_count = positional.len();
        positional.extend(args);
        if defaults.len() > positional.len() {
            return Err(crate::PyError::value_error(
                "more positional defaults than args on arguments",
            ));
        }
        let first_default = positional.len() - defaults.len();
        let mut positional: Vec<ast::ParameterWithDefault> = positional
            .into_iter()
            .map(|parameter| ast::ParameterWithDefault {
                range: Default::default(),
                node_index: Default::default(),
                parameter,
                default: None,
            })
            .collect();
        for (offset, default) in defaults.into_iter().enumerate() {
            positional[first_default + offset].default = Some(Box::new(default));
        }
        let args = positional.split_off(posonly_count);
        let posonlyargs = positional;
        let mut kwonly: Vec<ast::ParameterWithDefault> = kwonlyargs
            .into_iter()
            .map(|parameter| ast::ParameterWithDefault {
                range: Default::default(),
                node_index: Default::default(),
                parameter,
                default: None,
            })
            .collect();
        for (index, default) in kw_defaults.into_iter().enumerate() {
            if unsafe { pyre_object::is_none(default) } {
                continue;
            }
            kwonly[index].default = Some(Box::new(self.recurse(|this| this.expr(default))?));
        }
        Ok(ast::Parameters {
            range: Default::default(),
            node_index: Default::default(),
            posonlyargs,
            args,
            vararg,
            kwonlyargs: kwonly,
            kwarg,
            runtime_defaults: None,
        })
    }

    fn parameter_list(
        &mut self,
        object: PyObjectRef,
        field: &str,
    ) -> AstResult<Vec<ast::Parameter>> {
        self.list(object, field, "arguments")?
            .into_iter()
            .map(|value| self.recurse(|this| this.parameter(value)))
            .collect()
    }

    fn opt_parameter(
        &mut self,
        object: PyObjectRef,
        field: &str,
    ) -> AstResult<Option<Box<ast::Parameter>>> {
        self.optional_field(object, field)?
            .map(|value| self.recurse(|this| this.parameter(value)).map(Box::new))
            .transpose()
    }

    fn parameter(&mut self, object: PyObjectRef) -> AstResult<ast::Parameter> {
        Ok(ast::Parameter {
            range: Default::default(),
            node_index: Default::default(),
            name: self.identifier(object, "arg", "arg")?,
            annotation: self.opt_expr(object, "annotation")?,
            runtime_type_comment: None,
            runtime_type_comment_bytes: None,
        })
    }

    fn with_item(&mut self, object: PyObjectRef) -> AstResult<ast::WithItem> {
        let context_expr = self.req_expr(object, "context_expr", "withitem")?;
        Ok(ast::WithItem {
            range: Default::default(),
            node_index: Default::default(),
            context_expr: *context_expr,
            optional_vars: self.opt_expr(object, "optional_vars")?,
        })
    }

    fn handler(&mut self, object: PyObjectRef) -> AstResult<ast::ExceptHandler> {
        if !self.is_node(object, "ExceptHandler")? {
            return Err(crate::PyError::type_error(format!(
                "expected some sort of excepthandler, but got {}",
                unsafe { pyre_object::type_name_of(object) }
            )));
        }
        Ok(ast::ExceptHandler::ExceptHandler(
            ast::ExceptHandlerExceptHandler {
                range: Default::default(),
                node_index: Default::default(),
                type_: self.opt_expr(object, "type")?,
                name: self.opt_identifier(object, "name")?,
                body: self.body(object, "body", "ExceptHandler")?,
                runtime_body: None,
            },
        ))
    }

    fn comprehension(&mut self, object: PyObjectRef) -> AstResult<ast::Comprehension> {
        let target = self.req_expr(object, "target", "comprehension")?;
        let iter = self.req_expr(object, "iter", "comprehension")?;
        let ifs = self.exprs(object, "ifs", "comprehension")?;
        let is_async = self.int_field(object, "is_async", "comprehension")?;
        Ok(ast::Comprehension {
            range: Default::default(),
            node_index: Default::default(),
            target: *target,
            iter: *iter,
            ifs,
            is_async: is_async != 0,
            runtime_ifs: None,
            runtime_is_async: None,
        })
    }

    fn comprehensions(
        &mut self,
        object: PyObjectRef,
        node: &str,
    ) -> AstResult<Vec<ast::Comprehension>> {
        self.list(object, "generators", node)?
            .into_iter()
            .map(|value| self.recurse(|this| this.comprehension(value)))
            .collect()
    }

    fn aliases(&mut self, object: PyObjectRef, node: &str) -> AstResult<Vec<ast::Alias>> {
        self.list(object, "names", node)?
            .into_iter()
            .map(|value| {
                Ok(ast::Alias {
                    range: Default::default(),
                    node_index: Default::default(),
                    name: self.identifier(value, "name", "alias")?,
                    asname: self.opt_identifier(value, "asname")?,
                })
            })
            .collect()
    }

    fn decorators(&mut self, object: PyObjectRef, node: &str) -> AstResult<Vec<ast::Decorator>> {
        self.list(object, "decorator_list", node)?
            .into_iter()
            .map(|value| {
                self.require_node(value, "expression")?;
                Ok(ast::Decorator {
                    range: Default::default(),
                    node_index: Default::default(),
                    expression: self.recurse(|this| this.expr(value))?,
                })
            })
            .collect()
    }

    /// `type_params` postdates the original positional constructors, so a
    /// manually built legacy node without the field means an empty list.
    fn type_params(
        &mut self,
        object: PyObjectRef,
        node: &str,
    ) -> AstResult<Option<Box<ast::TypeParams>>> {
        let Some(field) = self.optional_field(object, "type_params")? else {
            return Ok(None);
        };
        if !unsafe { pyre_object::is_list(field) } {
            return Err(crate::PyError::type_error(format!(
                "{node} field \"type_params\" must be a list, not a {}",
                class_name(field)
            )));
        }
        let values = unsafe { pyre_object::w_list_items_copy_as_vec(field) };
        if values.is_empty() {
            return Ok(None);
        }
        let type_params = values
            .into_iter()
            .map(|value| self.recurse(|this| this.type_param(value)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Box::new(ast::TypeParams {
            range: Default::default(),
            node_index: Default::default(),
            type_params,
            runtime_type_params: None,
        })))
    }

    fn type_param(&mut self, object: PyObjectRef) -> AstResult<ast::TypeParam> {
        if self.is_node(object, "TypeVar")? {
            let name = self.identifier(object, "name", "TypeVar")?;
            Ok(ast::TypeParam::TypeVar(ast::TypeParamTypeVar {
                node_index: Default::default(),
                range: Default::default(),
                name,
                bound: self.opt_expr(object, "bound")?,
                default: self.opt_expr(object, "default_value")?,
            }))
        } else if self.is_node(object, "TypeVarTuple")? {
            let name = self.identifier(object, "name", "TypeVarTuple")?;
            Ok(ast::TypeParam::TypeVarTuple(ast::TypeParamTypeVarTuple {
                node_index: Default::default(),
                range: Default::default(),
                name,
                default: self.opt_expr(object, "default_value")?,
            }))
        } else if self.is_node(object, "ParamSpec")? {
            let name = self.identifier(object, "name", "ParamSpec")?;
            Ok(ast::TypeParam::ParamSpec(ast::TypeParamParamSpec {
                node_index: Default::default(),
                range: Default::default(),
                name,
                default: self.opt_expr(object, "default_value")?,
            }))
        } else {
            Err(crate::PyError::type_error(format!(
                "expected some sort of type_param, but got {}",
                unsafe { pyre_object::type_name_of(object) }
            )))
        }
    }

    fn match_case(&mut self, object: PyObjectRef) -> AstResult<ast::MatchCase> {
        let pattern = self.field(object, "pattern", "match_case")?;
        let pattern = self.recurse(|this| this.pattern(pattern))?;
        Ok(ast::MatchCase {
            range: Default::default(),
            node_index: Default::default(),
            pattern,
            guard: self.opt_expr(object, "guard")?,
            body: self.body(object, "body", "match_case")?,
            runtime_body: None,
        })
    }

    fn patterns(
        &mut self,
        object: PyObjectRef,
        field: &str,
        node: &str,
    ) -> AstResult<Vec<ast::Pattern>> {
        self.list(object, field, node)?
            .into_iter()
            .map(|value| self.recurse(|this| this.pattern(value)))
            .collect()
    }

    fn pattern(&mut self, object: PyObjectRef) -> AstResult<ast::Pattern> {
        if self.is_node(object, "MatchValue")? {
            Ok(ast::Pattern::MatchValue(ast::PatternMatchValue {
                node_index: Default::default(),
                range: Default::default(),
                value: self.req_expr(object, "value", "MatchValue")?,
            }))
        } else if self.is_node(object, "MatchSingleton")? {
            let value = self.field(object, "value", "MatchSingleton")?;
            let value = unsafe {
                if pyre_object::is_none(value) {
                    ast::Singleton::None
                } else if pyre_object::is_bool(value) {
                    if pyre_object::w_bool_get_value(value) {
                        ast::Singleton::True
                    } else {
                        ast::Singleton::False
                    }
                } else {
                    return Err(crate::PyError::value_error(
                        "MatchSingleton can only contain True, False and None",
                    ));
                }
            };
            Ok(ast::Pattern::MatchSingleton(ast::PatternMatchSingleton {
                node_index: Default::default(),
                range: Default::default(),
                value,
            }))
        } else if self.is_node(object, "MatchSequence")? {
            Ok(ast::Pattern::MatchSequence(ast::PatternMatchSequence {
                node_index: Default::default(),
                range: Default::default(),
                patterns: self.patterns(object, "patterns", "MatchSequence")?,
                runtime_patterns: None,
            }))
        } else if self.is_node(object, "MatchMapping")? {
            let keys = self.exprs(object, "keys", "MatchMapping")?;
            let patterns = self.patterns(object, "patterns", "MatchMapping")?;
            Ok(ast::Pattern::MatchMapping(ast::PatternMatchMapping {
                node_index: Default::default(),
                range: Default::default(),
                keys,
                patterns,
                rest: self.opt_identifier(object, "rest")?,
                runtime_keys: None,
                runtime_patterns: None,
            }))
        } else if self.is_node(object, "MatchClass")? {
            let cls = self.req_expr(object, "cls", "MatchClass")?;
            let patterns = self.patterns(object, "patterns", "MatchClass")?;
            let attrs = self.identifiers(object, "kwd_attrs", "MatchClass")?;
            let kwd_patterns = self.patterns(object, "kwd_patterns", "MatchClass")?;
            if attrs.len() != kwd_patterns.len() {
                return Err(crate::PyError::value_error(
                    "MatchClass doesn't have the same number of keyword attributes as patterns",
                ));
            }
            let keywords = attrs
                .into_iter()
                .zip(kwd_patterns)
                .map(|(attr, pattern)| ast::PatternKeyword {
                    range: Default::default(),
                    node_index: Default::default(),
                    attr,
                    pattern,
                })
                .collect();
            Ok(ast::Pattern::MatchClass(ast::PatternMatchClass {
                node_index: Default::default(),
                range: Default::default(),
                cls,
                arguments: ast::PatternArguments {
                    range: Default::default(),
                    node_index: Default::default(),
                    patterns,
                    keywords,
                },
                runtime_patterns: None,
                runtime_kwd_attrs: None,
                runtime_kwd_patterns: None,
            }))
        } else if self.is_node(object, "MatchStar")? {
            Ok(ast::Pattern::MatchStar(ast::PatternMatchStar {
                node_index: Default::default(),
                range: Default::default(),
                name: self.opt_identifier(object, "name")?,
            }))
        } else if self.is_node(object, "MatchAs")? {
            let pattern = self
                .optional_field(object, "pattern")?
                .map(|value| self.recurse(|this| this.pattern(value)).map(Box::new))
                .transpose()?;
            Ok(ast::Pattern::MatchAs(ast::PatternMatchAs {
                node_index: Default::default(),
                range: Default::default(),
                pattern,
                name: self.opt_identifier(object, "name")?,
            }))
        } else if self.is_node(object, "MatchOr")? {
            Ok(ast::Pattern::MatchOr(ast::PatternMatchOr {
                node_index: Default::default(),
                range: Default::default(),
                patterns: self.patterns(object, "patterns", "MatchOr")?,
                runtime_patterns: None,
            }))
        } else {
            Err(crate::PyError::type_error(format!(
                "expected some sort of pattern, but got {}",
                unsafe { pyre_object::type_name_of(object) }
            )))
        }
    }

    /// `_validate_stmts` / `_validate_exprs` (validate.py:132, :151) reject a
    /// missing element of a statement or expression list.  The compiler AST
    /// has nowhere to hold one, so unlike the rest of the validation this runs
    /// during the conversion, where it is still visible.  Lists of the other
    /// ASDL nodes carry no such check.
    fn require_node(&self, value: PyObjectRef, kind: &str) -> AstResult<()> {
        if unsafe { pyre_object::is_none(value) } {
            return Err(crate::PyError::value_error(format!(
                "None disallowed in {kind} list"
            )));
        }
        Ok(())
    }

    fn body(&mut self, object: PyObjectRef, field: &str, node: &str) -> AstResult<Vec<ast::Stmt>> {
        self.list(object, field, node)?
            .into_iter()
            .map(|value| {
                self.require_node(value, "statement")?;
                self.recurse(|this| this.stmt(value))
            })
            .collect()
    }

    fn exprs(&mut self, object: PyObjectRef, field: &str, node: &str) -> AstResult<Vec<ast::Expr>> {
        self.list(object, field, node)?
            .into_iter()
            .map(|value| {
                self.require_node(value, "expression")?;
                self.recurse(|this| this.expr(value))
            })
            .collect()
    }

    fn req_expr(
        &mut self,
        object: PyObjectRef,
        field: &str,
        node: &str,
    ) -> AstResult<Box<ast::Expr>> {
        let value = self.field(object, field, node)?;
        Ok(Box::new(self.recurse(|this| this.expr(value))?))
    }

    fn opt_expr(&mut self, object: PyObjectRef, field: &str) -> AstResult<Option<Box<ast::Expr>>> {
        self.optional_field(object, field)?
            .map(|value| self.recurse(|this| this.expr(value)).map(Box::new))
            .transpose()
    }

    fn identifier(
        &self,
        object: PyObjectRef,
        field: &str,
        node: &str,
    ) -> AstResult<ast::Identifier> {
        Ok(ast::Identifier::new(
            self.string(object, field, node)?,
            Default::default(),
        ))
    }

    fn opt_identifier(
        &self,
        object: PyObjectRef,
        field: &str,
    ) -> AstResult<Option<ast::Identifier>> {
        let Some(value) = self.optional_field(object, field)? else {
            return Ok(None);
        };
        if !unsafe { pyre_object::is_str(value) } {
            return Err(crate::PyError::type_error(
                "AST identifier must be of type str",
            ));
        }
        Ok(Some(ast::Identifier::new(
            unsafe { pyre_object::w_str_get_value(value).to_string() },
            Default::default(),
        )))
    }

    fn identifiers(
        &self,
        object: PyObjectRef,
        field: &str,
        node: &str,
    ) -> AstResult<Vec<ast::Identifier>> {
        self.list(object, field, node)?
            .into_iter()
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
            .collect()
    }

    fn int_field(&self, object: PyObjectRef, field: &str, node: &str) -> AstResult<i64> {
        let value = self.field(object, field, node)?;
        crate::builtins::space_index_w(value)
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
                .map(|arg| {
                    self.require_node(arg, "expression")?;
                    self.recurse(|this| this.expr(arg))
                })
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
                .map(|element| {
                    self.require_node(element, "expression")?;
                    self.recurse(|this| this.expr(element))
                })
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
        } else if self.is_node(object, "BoolOp")? {
            let op = self.field(object, "op", "BoolOp")?;
            Ok(ast::Expr::BoolOp(ast::ExprBoolOp {
                node_index: Default::default(),
                range: Default::default(),
                op: self.boolop(op)?,
                values: self.exprs(object, "values", "BoolOp")?,
                runtime_values: None,
            }))
        } else if self.is_node(object, "NamedExpr")? {
            let target = self.req_expr(object, "target", "NamedExpr")?;
            let value = self.req_expr(object, "value", "NamedExpr")?;
            Ok(ast::Expr::Named(ast::ExprNamed {
                node_index: Default::default(),
                range: Default::default(),
                target,
                value,
            }))
        } else if self.is_node(object, "Lambda")? {
            let args = self.field(object, "args", "Lambda")?;
            let parameters = self.recurse(|this| this.parameters(args))?;
            let body = self.req_expr(object, "body", "Lambda")?;
            Ok(ast::Expr::Lambda(ast::ExprLambda {
                node_index: Default::default(),
                range: Default::default(),
                parameters: Some(Box::new(parameters)),
                body,
            }))
        } else if self.is_node(object, "IfExp")? {
            let test = self.req_expr(object, "test", "IfExp")?;
            let body = self.req_expr(object, "body", "IfExp")?;
            let orelse = self.req_expr(object, "orelse", "IfExp")?;
            Ok(ast::Expr::If(ast::ExprIf {
                node_index: Default::default(),
                range: Default::default(),
                test,
                body,
                orelse,
            }))
        } else if self.is_node(object, "Dict")? {
            let keys = self.list(object, "keys", "Dict")?;
            let values = self.list(object, "values", "Dict")?;
            if keys.len() != values.len() {
                return Err(crate::PyError::value_error(
                    "Dict doesn't have the same number of keys as values",
                ));
            }
            // A `None` key is the `**mapping` spread, which has no key node.
            let items = keys
                .into_iter()
                .zip(values)
                .map(|(key, value)| {
                    let key = if unsafe { pyre_object::is_none(key) } {
                        None
                    } else {
                        Some(self.recurse(|this| this.expr(key))?)
                    };
                    self.require_node(value, "expression")?;
                    Ok(ast::DictItem {
                        key,
                        value: self.recurse(|this| this.expr(value))?,
                    })
                })
                .collect::<Result<Vec<_>, crate::PyError>>()?;
            Ok(ast::Expr::Dict(ast::ExprDict {
                node_index: Default::default(),
                range: Default::default(),
                items,
                runtime_values: None,
            }))
        } else if self.is_node(object, "Set")? {
            Ok(ast::Expr::Set(ast::ExprSet {
                node_index: Default::default(),
                range: Default::default(),
                elts: self.exprs(object, "elts", "Set")?,
                runtime_elts: None,
            }))
        } else if self.is_node(object, "ListComp")? {
            let elt = self.req_expr(object, "elt", "ListComp")?;
            let generators = self.comprehensions(object, "ListComp")?;
            Ok(ast::Expr::ListComp(ast::ExprListComp {
                node_index: Default::default(),
                range: Default::default(),
                elt,
                generators,
            }))
        } else if self.is_node(object, "SetComp")? {
            let elt = self.req_expr(object, "elt", "SetComp")?;
            let generators = self.comprehensions(object, "SetComp")?;
            Ok(ast::Expr::SetComp(ast::ExprSetComp {
                node_index: Default::default(),
                range: Default::default(),
                elt,
                generators,
            }))
        } else if self.is_node(object, "DictComp")? {
            let key = self.req_expr(object, "key", "DictComp")?;
            let value = self.req_expr(object, "value", "DictComp")?;
            let generators = self.comprehensions(object, "DictComp")?;
            Ok(ast::Expr::DictComp(ast::ExprDictComp {
                node_index: Default::default(),
                range: Default::default(),
                key,
                value,
                generators,
            }))
        } else if self.is_node(object, "GeneratorExp")? {
            let elt = self.req_expr(object, "elt", "GeneratorExp")?;
            let generators = self.comprehensions(object, "GeneratorExp")?;
            Ok(ast::Expr::Generator(ast::ExprGenerator {
                node_index: Default::default(),
                range: Default::default(),
                elt,
                generators,
                parenthesized: true,
            }))
        } else if self.is_node(object, "Await")? {
            Ok(ast::Expr::Await(ast::ExprAwait {
                node_index: Default::default(),
                range: Default::default(),
                value: self.req_expr(object, "value", "Await")?,
            }))
        } else if self.is_node(object, "Yield")? {
            Ok(ast::Expr::Yield(ast::ExprYield {
                node_index: Default::default(),
                range: Default::default(),
                value: self.opt_expr(object, "value")?,
            }))
        } else if self.is_node(object, "YieldFrom")? {
            Ok(ast::Expr::YieldFrom(ast::ExprYieldFrom {
                node_index: Default::default(),
                range: Default::default(),
                value: self.req_expr(object, "value", "YieldFrom")?,
            }))
        } else if self.is_node(object, "Compare")? {
            let left = self.req_expr(object, "left", "Compare")?;
            let ops = self
                .list(object, "ops", "Compare")?
                .into_iter()
                .map(|op| self.cmpop(op))
                .collect::<Result<Vec<_>, _>>()?;
            let comparators = self.exprs(object, "comparators", "Compare")?;
            Ok(ast::Expr::Compare(ast::ExprCompare {
                node_index: Default::default(),
                range: Default::default(),
                left,
                ops: ops.into_boxed_slice(),
                comparators: comparators.into_boxed_slice(),
                runtime_comparators: None,
            }))
        } else if self.is_node(object, "Subscript")? {
            let value = self.req_expr(object, "value", "Subscript")?;
            let slice = self.req_expr(object, "slice", "Subscript")?;
            let ctx = self.context(self.field(object, "ctx", "Subscript")?)?;
            Ok(ast::Expr::Subscript(ast::ExprSubscript {
                node_index: Default::default(),
                range: Default::default(),
                value,
                slice,
                ctx,
            }))
        } else if self.is_node(object, "Starred")? {
            let value = self.req_expr(object, "value", "Starred")?;
            let ctx = self.context(self.field(object, "ctx", "Starred")?)?;
            Ok(ast::Expr::Starred(ast::ExprStarred {
                node_index: Default::default(),
                range: Default::default(),
                value,
                ctx,
            }))
        } else if self.is_node(object, "Slice")? {
            Ok(ast::Expr::Slice(ast::ExprSlice {
                node_index: Default::default(),
                range: Default::default(),
                lower: self.opt_expr(object, "lower")?,
                upper: self.opt_expr(object, "upper")?,
                step: self.opt_expr(object, "step")?,
            }))
        } else if self.is_node(object, "JoinedStr")? {
            let values = self.exprs(object, "values", "JoinedStr")?;
            Ok(fstring(Vec::new(), Some(values)))
        } else if self.is_node(object, "FormattedValue")? {
            let element = self.interpolation(object)?;
            Ok(fstring(vec![element], None))
        } else {
            Err(crate::PyError::type_error(format!(
                "expected some sort of expr, but got {}",
                unsafe { pyre_object::type_name_of(object) }
            )))
        }
    }

    fn interpolation(&mut self, object: PyObjectRef) -> AstResult<ast::InterpolatedStringElement> {
        let expression = self.req_expr(object, "value", "FormattedValue")?;
        let conversion = self.conversion(object)?;
        let format_spec = self.opt_expr(object, "format_spec")?;
        Ok(ast::InterpolatedStringElement::Interpolation(
            ast::InterpolatedElement {
                node_index: Default::default(),
                range: Default::default(),
                expression,
                debug_text: None,
                conversion,
                // A spec is an expression that gets compiled like any other
                // (`visit_FormattedValue`, codegen.py:2371). The compiler AST
                // instead keeps the elements the parser found between the
                // braces, which an object does not carry, so the spec rides
                // the field below and the parsed shape stays empty.
                format_spec: None,
                runtime_str: None,
                runtime_interpolation_format_spec: None,
                runtime_formatted_value_format_spec: format_spec,
            },
        ))
    }

    /// `visit_FormattedValue` (codegen.py:2364) matches `s`, `r` and `a` and
    /// leaves anything else at no conversion at all; 3.14 stops instead, so a
    /// character it does not know is an error here.
    fn conversion(&self, object: PyObjectRef) -> AstResult<ast::ConversionFlag> {
        match self.int_field(object, "conversion", "FormattedValue")? {
            -1 => Ok(ast::ConversionFlag::None),
            value if value == i64::from(b's') => Ok(ast::ConversionFlag::Str),
            value if value == i64::from(b'r') => Ok(ast::ConversionFlag::Repr),
            value if value == i64::from(b'a') => Ok(ast::ConversionFlag::Ascii),
            value => Err(crate::PyError::system_error(format!(
                "Unrecognized conversion character {value}"
            ))),
        }
    }

    fn boolop(&self, object: PyObjectRef) -> AstResult<ast::BoolOp> {
        for (name, op) in [("And", ast::BoolOp::And), ("Or", ast::BoolOp::Or)] {
            if self.is_node(object, name)? {
                return Ok(op);
            }
        }
        Err(crate::PyError::type_error("expected some sort of boolop"))
    }

    fn cmpop(&self, object: PyObjectRef) -> AstResult<ast::CmpOp> {
        for (name, op) in [
            ("Eq", ast::CmpOp::Eq),
            ("NotEq", ast::CmpOp::NotEq),
            ("Lt", ast::CmpOp::Lt),
            ("LtE", ast::CmpOp::LtE),
            ("Gt", ast::CmpOp::Gt),
            ("GtE", ast::CmpOp::GtE),
            ("Is", ast::CmpOp::Is),
            ("IsNot", ast::CmpOp::IsNot),
            ("In", ast::CmpOp::In),
            ("NotIn", ast::CmpOp::NotIn),
        ] {
            if self.is_node(object, name)? {
                return Ok(op);
            }
        }
        Err(crate::PyError::type_error("expected some sort of cmpop"))
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
            } else if pyre_object::is_complex(object) {
                Ok(ast::ConstantValue::Complex {
                    real: pyre_object::w_complex_get_real(object),
                    imag: pyre_object::w_complex_get_imag(object),
                })
            } else if pyre_object::is_ellipsis(object) {
                Ok(ast::ConstantValue::Ellipsis)
            } else if pyre_object::is_tuple(object) {
                // A container constant never comes out of the parser; it reaches
                // here from a tree an optimizer folded, and it nests. The node
                // depth guard does not apply: it counts AST nodes, and a
                // constant nested past it still compiles where 3.14 compiles.
                Ok(ast::ConstantValue::Tuple(
                    pyre_object::w_tuple_items_copy_as_vec(object)
                        .into_iter()
                        .map(|item| self.constant_value(item))
                        .collect::<Result<Vec<_>, _>>()?,
                ))
            } else if pyre_object::is_frozenset(object) {
                Ok(ast::ConstantValue::Frozenset(
                    pyre_object::w_set_items(object)
                        .into_iter()
                        .map(|item| self.constant_value(item))
                        .collect::<Result<Vec<_>, _>>()?,
                ))
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

/// An f-string expression built from `_ast` objects.  `visit_JoinedStr`
/// (codegen.py:2357) compiles the values one after another and joins them,
/// which is the shape an object carries; the compiler AST instead holds the
/// literal and interpolated parts the parser split, so the values are handed
/// over as the parts to join and that part list stays empty.  A
/// `FormattedValue` on its own is a single interpolated part, which does fit
/// the part list.
fn fstring(
    elements: Vec<ast::InterpolatedStringElement>,
    runtime_joined_str: Option<Vec<ast::Expr>>,
) -> ast::Expr {
    ast::Expr::FString(ast::ExprFString {
        node_index: Default::default(),
        range: Default::default(),
        value: ast::FStringValue::single(ast::FString {
            node_index: Default::default(),
            range: Default::default(),
            elements: elements.into(),
            flags: ast::FStringFlags::empty(),
        }),
        runtime_joined_str,
        runtime_values: None,
    })
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

        let mut parts = Vec::new();
        for part in node.value.iter() {
            match part {
                ast::FStringPart::Literal(literal) => {
                    push_literal(&mut parts, range(literal.range), &literal.value)
                }
                ast::FStringPart::FString(fstring) => {
                    self.interpolated_elements(&fstring.elements, &mut parts)?
                }
            }
        }
        let values = self.joined_values(parts)?;
        self.node(
            "JoinedStr",
            Some(range(node.range)),
            &[("values", self.list(values))],
        )
    }

    fn joined_values(&self, parts: Vec<JoinedPart>) -> Result<Vec<PyObjectRef>, crate::PyError> {
        parts
            .into_iter()
            .map(|part| match part {
                JoinedPart::Literal { start, end, value } => {
                    self.constant((start, end), self.string(&value), pyre_object::w_none())
                }
                JoinedPart::Value(value) => Ok(value),
            })
            .collect()
    }

    fn interpolated_elements(
        &self,
        elements: &[ast::InterpolatedStringElement],
        parts: &mut Vec<JoinedPart>,
    ) -> Result<(), crate::PyError> {
        for element in elements {
            match element {
                ast::InterpolatedStringElement::Literal(literal) => {
                    push_literal(parts, range(literal.range), &literal.value)
                }
                ast::InterpolatedStringElement::Interpolation(interpolation) => {
                    let mut conversion = interpolation.conversion;
                    if let Some(debug_text) = interpolation.debug_text.as_ref() {
                        self.push_debug_text(parts, interpolation, debug_text);
                        // fstring.py:315 — a debugging expression defaults to
                        // `!r` when neither a conversion nor a format spec
                        // says how to render the value.
                        if matches!(
                            (conversion, &interpolation.format_spec),
                            (ast::ConversionFlag::None, None)
                        ) {
                            conversion = ast::ConversionFlag::Repr;
                        }
                    }
                    let format_spec = interpolation
                        .format_spec
                        .as_deref()
                        .map(|spec| {
                            let mut spec_parts = Vec::new();
                            self.interpolated_elements(&spec.elements, &mut spec_parts)?;
                            let values = self.joined_values(spec_parts)?;
                            self.node(
                                "JoinedStr",
                                Some(range(spec.range)),
                                &[("values", self.list(values))],
                            )
                        })
                        .transpose()?;
                    parts.push(JoinedPart::Value(self.node(
                        "FormattedValue",
                        Some(range(interpolation.range)),
                        &[
                            ("value", self.expr(&interpolation.expression)?),
                            (
                                "conversion",
                                pyre_object::w_int_new(conversion as i8 as i64),
                            ),
                            ("format_spec", self.optional(format_spec)),
                        ],
                    )?));
                }
            }
        }
        Ok(())
    }

    /// The literal an `=` conversion echoes.  `fstring_find_expr`
    /// (fstring.py:279) takes it as one slice of the source, from the start of
    /// the expression through the `=` and the whitespace after it; the parser
    /// here hands over that frame as the text on either side of the
    /// expression, so the slice is put back together from the two.
    fn push_debug_text(
        &self,
        parts: &mut Vec<JoinedPart>,
        interpolation: &ast::InterpolatedElement,
        debug_text: &ast::DebugText,
    ) {
        use ruff_text_size::Ranged;
        let (start, end) = range(interpolation.expression.range());
        let leading = strip_debug_comments(&debug_text.leading);
        let trailing = strip_debug_comments(&debug_text.trailing);
        let expression = &self.source[start as usize..end as usize];
        push_literal(
            parts,
            (
                start.saturating_sub(leading.len() as u32),
                end + trailing.len() as u32,
            ),
            &[leading.as_str(), expression, trailing.as_str()].concat(),
        );
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
                let int_type = crate::typedef::gettypefor(&pyre_object::INT_TYPE)
                    .map_or(PY_NULL, |p| p.as_ptr());
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
                let int_type = crate::typedef::gettypefor(&pyre_object::INT_TYPE)
                    .map_or(PY_NULL, |p| p.as_ptr());
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

/// `%T`-style class name of an AST object.  The `_ast` node types are heap
/// types, so the name sits on the class the instance points at rather than on
/// the layout its `ob_type` names.
fn class_name(object: PyObjectRef) -> &'static str {
    match crate::typedef::r#type(object) {
        Some(w_type) => unsafe { pyre_object::w_type_get_name(w_type.as_ptr()) },
        None => unsafe { pyre_object::type_name_of(object) },
    }
}

/// A value of a `JoinedStr` under construction.  `add_constant_string`
/// (fstring.py:23) folds a piece into the `Constant` before it, keeping that
/// node's start and taking the new end, and an empty piece is dropped
/// (`f_string_to_ast_node`, fstring.py:449); what the parser kept apart -- an
/// implicit concatenation, the text an `=` conversion echoes -- therefore
/// reaches the tree as one node, not one per piece.
enum JoinedPart {
    Literal { start: u32, end: u32, value: String },
    Value(PyObjectRef),
}

fn push_literal(parts: &mut Vec<JoinedPart>, (start, end): (u32, u32), value: &str) {
    if value.is_empty() {
        return;
    }
    if let Some(JoinedPart::Literal {
        end: last_end,
        value: last,
        ..
    }) = parts.last_mut()
    {
        *last_end = end;
        last.push_str(value);
        return;
    }
    parts.push(JoinedPart::Literal {
        start,
        end,
        value: value.to_string(),
    });
}

/// A comment inside the braces runs to the end of the line and is no part of
/// the text an `=` conversion echoes.
fn strip_debug_comments(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_comment = false;
    for ch in text.chars() {
        if in_comment {
            if ch == '\n' {
                in_comment = false;
                result.push(ch);
            }
        } else if ch == '#' {
            in_comment = true;
        } else {
            result.push(ch);
        }
    }
    result
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
