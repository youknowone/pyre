//! _ast implementation — PyPy: pypy/module/_ast/moduledef.py +
//! pypy/interpreter/astcompiler/ast.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use pyre_object::PyObjectRef;

const LOCATION_ATTRIBUTES: &[&str] = &["lineno", "col_offset", "end_lineno", "end_col_offset"];

fn tuple_of_names(names: &[&str]) -> PyObjectRef {
    pyre_object::w_tuple_new(names.iter().map(|name| pyre_object::w_str_new(name)).collect())
}

fn ast_init(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let Some((&zelf, values)) = positional.split_first() else {
        return Err(crate::PyError::type_error("AST.__init__() missing self"));
    };
    let fields_obj = crate::baseobjspace::getattr_str(zelf, "_fields")?;
    let fields = unsafe { pyre_object::w_tuple_items_copy_as_vec(fields_obj) };
    if values.len() > fields.len() {
        return Err(crate::PyError::type_error(format!(
            "{} constructor takes at most {} positional arguments",
            unsafe { pyre_object::type_name_of(zelf) },
            fields.len()
        )));
    }
    let mut positional_names = Vec::with_capacity(values.len());
    for (&field, &value) in fields.iter().zip(values) {
        let name = unsafe { pyre_object::w_str_get_value(field) };
        positional_names.push(name.to_owned());
        crate::baseobjspace::setattr_str(zelf, name, value)?;
    }
    if let Some(kwargs) = kwargs {
        for (name, value) in unsafe { pyre_object::w_dict_str_entries(kwargs) } {
            if name == "__pyre_kw__" {
                continue;
            }
            if positional_names.iter().any(|positional| positional == &name) {
                return Err(crate::PyError::type_error(format!(
                    "{} got multiple values for argument '{name}'",
                    unsafe { pyre_object::type_name_of(zelf) }
                )));
            }
            crate::baseobjspace::setattr_str(zelf, &name, value)?;
        }
    }
    Ok(pyre_object::w_none())
}

fn node_fields(name: &str) -> &'static [&'static str] {
    match name {
        "Module" => &["body", "type_ignores"],
        "Interactive" => &["body"],
        "Expression" => &["body"],
        "FunctionType" => &["argtypes", "returns"],
        "FunctionDef" | "AsyncFunctionDef" => &["name", "args", "body", "decorator_list", "returns", "type_comment", "type_params"],
        "ClassDef" => &["name", "bases", "keywords", "body", "decorator_list", "type_params"],
        "Return" => &["value"],
        "Delete" => &["targets"],
        "Assign" => &["targets", "value", "type_comment"],
        "TypeAlias" => &["name", "type_params", "value"],
        "AugAssign" => &["target", "op", "value"],
        "AnnAssign" => &["target", "annotation", "value", "simple"],
        "For" | "AsyncFor" => &["target", "iter", "body", "orelse", "type_comment"],
        "While" | "If" => &["test", "body", "orelse"],
        "With" | "AsyncWith" => &["items", "body", "type_comment"],
        "Match" => &["subject", "cases"],
        "Raise" => &["exc", "cause"],
        "Try" | "TryStar" => &["body", "handlers", "orelse", "finalbody"],
        "Assert" => &["test", "msg"],
        "Import" => &["names"],
        "ImportFrom" => &["module", "names", "level"],
        "Global" | "Nonlocal" => &["names"],
        "Expr" => &["value"],
        "Pass" | "Break" | "Continue" => &[],
        "BoolOp" => &["op", "values"],
        "NamedExpr" => &["target", "value"],
        "BinOp" => &["left", "op", "right"],
        "UnaryOp" => &["op", "operand"],
        "Lambda" => &["args", "body"],
        "IfExp" => &["test", "body", "orelse"],
        "Dict" => &["keys", "values"],
        "Set" => &["elts"],
        "ListComp" | "SetComp" | "GeneratorExp" => &["elt", "generators"],
        "DictComp" => &["key", "value", "generators"],
        "Await" | "Yield" | "YieldFrom" => &["value"],
        "Compare" => &["left", "ops", "comparators"],
        "Call" => &["func", "args", "keywords"],
        "FormattedValue" => &["value", "conversion", "format_spec"],
        "JoinedStr" => &["values"],
        "Constant" => &["value", "kind"],
        "Attribute" => &["value", "attr", "ctx"],
        "Subscript" => &["value", "slice", "ctx"],
        "Starred" => &["value", "ctx"],
        "Name" => &["id", "ctx"],
        "List" | "Tuple" => &["elts", "ctx"],
        "Slice" => &["lower", "upper", "step"],
        "ExceptHandler" => &["type", "name", "body"],
        "MatchValue" => &["value"],
        "MatchSingleton" => &["value"],
        "MatchSequence" => &["patterns"],
        "MatchMapping" => &["keys", "patterns", "rest"],
        "MatchClass" => &["cls", "patterns", "kwd_attrs", "kwd_patterns"],
        "MatchStar" => &["name"],
        "MatchAs" => &["pattern", "name"],
        "MatchOr" => &["patterns"],
        "TypeIgnore" => &["lineno", "tag"],
        "TypeVar" => &["name", "bound", "default_value"],
        "ParamSpec" | "TypeVarTuple" => &["name", "default_value"],
        "comprehension" => &["target", "iter", "ifs", "is_async"],
        "arguments" => &["posonlyargs", "args", "vararg", "kwonlyargs", "kw_defaults", "kwarg", "defaults"],
        "arg" => &["arg", "annotation", "type_comment"],
        "keyword" => &["arg", "value"],
        "alias" => &["name", "asname"],
        "withitem" => &["context_expr", "optional_vars"],
        "match_case" => &["pattern", "guard", "body"],
        _ => &[],
    }
}

fn node_has_location(name: &str) -> bool {
    matches!(name, "FunctionDef" | "AsyncFunctionDef" | "ClassDef" | "Return" | "Delete" | "Assign" | "TypeAlias" | "AugAssign" | "AnnAssign" | "For" | "AsyncFor" | "While" | "If" | "With" | "AsyncWith" | "Match" | "Raise" | "Try" | "TryStar" | "Assert" | "Import" | "ImportFrom" | "Global" | "Nonlocal" | "Expr" | "Pass" | "Break" | "Continue" | "BoolOp" | "NamedExpr" | "BinOp" | "UnaryOp" | "Lambda" | "IfExp" | "Dict" | "Set" | "ListComp" | "SetComp" | "DictComp" | "GeneratorExp" | "Await" | "Yield" | "YieldFrom" | "Compare" | "Call" | "FormattedValue" | "JoinedStr" | "Constant" | "Attribute" | "Subscript" | "Starred" | "Name" | "List" | "Tuple" | "Slice" | "ExceptHandler" | "MatchValue" | "MatchSingleton" | "MatchSequence" | "MatchMapping" | "MatchClass" | "MatchStar" | "MatchAs" | "MatchOr" | "TypeVar" | "ParamSpec" | "TypeVarTuple" | "arg" | "keyword" | "alias")
}

/// _ast stub — PyPy: pypy/module/_ast/
///
/// Exposes the AST node type hierarchy. The node types are created as **heap
/// types** (via `type(name, bases, {})`) following the ASDL hierarchy
/// (`AST` → abstract group → concrete node), so `ast.py` can subclass them
/// (`class Suite(mod)`) and monkeypatch them (`Tuple.dims = property(...)`),
/// matching CPython where `_ast` types are heap types. Compiler-native Ruff
/// nodes are converted to instances of these public types by `convert.rs`.
pub fn register_module(ns: pyre_object::PyObjectRef) {
    // `type(name, (base,), {"__module__": "ast"})` — a fresh heap type. The
    // generated AST types report `__module__ == "ast"` (astcompiler/ast.py:150;
    // the host `_ast.Module.__module__` is likewise `'ast'`).
    let make = |name: &str, base: PyObjectRef| -> PyObjectRef {
        let dict = pyre_object::w_dict_new();
        crate::baseobjspace::setitem(dict, pyre_object::w_str_new("__module__"), pyre_object::w_str_new("ast"))
            .expect("set __module__ on _ast type namespace");
        crate::baseobjspace::setitem(
            dict,
            pyre_object::w_str_new("_fields"),
            tuple_of_names(node_fields(name)),
        )
        .expect("set _fields on _ast type namespace");
        crate::baseobjspace::setitem(
            dict,
            pyre_object::w_str_new("_attributes"),
            tuple_of_names(if node_has_location(name) {
                LOCATION_ATTRIBUTES
            } else {
                &[]
            }),
        )
        .expect("set _attributes on _ast type namespace");
        let field_types = pyre_object::w_dict_new();
        for field in node_fields(name) {
            crate::baseobjspace::setitem(
                field_types,
                pyre_object::w_str_new(field),
                crate::typedef::w_object(),
            )
            .expect("set _field_types item");
        }
        crate::baseobjspace::setitem(
            dict,
            pyre_object::w_str_new("_field_types"),
            field_types,
        )
        .expect("set _field_types on _ast type namespace");
        let args = [
            pyre_object::w_str_new(name),
            pyre_object::w_tuple_new(vec![base]),
            dict,
        ];
        crate::builtins::type_descr_new(&args).expect("_ast heap type creation")
    };

    // Root: AST(object).
    let ast = make("AST", crate::typedef::w_object());
    crate::baseobjspace::setattr_str(
        ast,
        "__init__",
        crate::make_builtin_function("__init__", ast_init),
    )
    .expect("set AST.__init__");
    crate::module_ns_store(ns, "AST", ast);

    // Abstract groups (direct AST subclasses) and their concrete members,
    // per the ASDL grammar.
    let groups: &[(&str, &[&str])] = &[
        ("mod", &["Module", "Interactive", "Expression", "FunctionType"]),
        (
            "stmt",
            &[
                "FunctionDef", "AsyncFunctionDef", "ClassDef", "Return", "Delete", "Assign",
                "TypeAlias", "AugAssign", "AnnAssign", "For", "AsyncFor", "While", "If", "With",
                "AsyncWith", "Match", "Raise", "Try", "TryStar", "Assert", "Import", "ImportFrom",
                "Global", "Nonlocal", "Expr", "Pass", "Break", "Continue",
            ],
        ),
        (
            "expr",
            &[
                "BoolOp", "NamedExpr", "BinOp", "UnaryOp", "Lambda", "IfExp", "Dict", "Set",
                "ListComp", "SetComp", "DictComp", "GeneratorExp", "Await", "Yield", "YieldFrom",
                "Compare", "Call", "FormattedValue", "JoinedStr", "Constant", "Attribute",
                "Subscript", "Starred", "Name", "List", "Tuple", "Slice",
            ],
        ),
        ("expr_context", &["Load", "Store", "Del"]),
        ("boolop", &["And", "Or"]),
        (
            "operator",
            &[
                "Add", "Sub", "Mult", "MatMult", "Div", "Mod", "Pow", "LShift", "RShift", "BitOr",
                "BitXor", "BitAnd", "FloorDiv",
            ],
        ),
        ("unaryop", &["Invert", "Not", "UAdd", "USub"]),
        ("cmpop", &["Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn"]),
        ("excepthandler", &["ExceptHandler"]),
        (
            "pattern",
            &[
                "MatchValue", "MatchSingleton", "MatchSequence", "MatchMapping", "MatchClass",
                "MatchStar", "MatchAs", "MatchOr",
            ],
        ),
        ("type_ignore", &["TypeIgnore"]),
        ("type_param", &["TypeVar", "ParamSpec", "TypeVarTuple"]),
    ];
    for (group, members) in groups {
        let g = make(group, ast);
        crate::module_ns_store(ns, group, g);
        for m in *members {
            let t = make(m, g);
            crate::module_ns_store(ns, m, t);
        }
    }

    // Leaf node types that are direct AST subclasses (no further subclasses).
    for name in &[
        "comprehension", "arguments", "arg", "keyword", "alias", "withitem", "match_case",
    ] {
        let t = make(name, ast);
        crate::module_ns_store(ns, name, t);
    }

    // `compile()` / `ast.parse()` flag bitmasks, used by `lib-python/3/ast.py`
    // (`flags = PyCF_ONLY_AST; flags |= PyCF_TYPE_COMMENTS`). Values mirror
    // `pypy/interpreter/astcompiler/consts.py:33-42`.
    for (name, value) in &[
        ("PyCF_ONLY_AST", 0x0400i64),
        ("PyCF_ALLOW_TOP_LEVEL_AWAIT", 0x2000),
        ("PyCF_TYPE_COMMENTS", 0x4000_0000),
        ("PyCF_OPTIMIZED_AST", 0x8000),
    ] {
        crate::module_ns_store(ns, name, pyre_object::w_int_new(*value));
    }
}
