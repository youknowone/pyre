//! _ast implementation — PyPy: pypy/module/_ast/moduledef.py +
//! pypy/interpreter/astcompiler/ast.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use pyre_object::PyObjectRef;

/// _ast stub — PyPy: pypy/module/_ast/
///
/// Exposes the AST node type hierarchy. The node types are created as **heap
/// types** (via `type(name, bases, {})`) following the ASDL hierarchy
/// (`AST` → abstract group → concrete node), so `ast.py` can subclass them
/// (`class Suite(mod)`) and monkeypatch them (`Tuple.dims = property(...)`),
/// matching CPython where `_ast` types are heap types. Actual AST node
/// construction is not supported because pyre uses RustPython's compiler.
pub fn register_module(ns: pyre_object::PyObjectRef) {
    // `type(name, (base,), {"__module__": "ast"})` — a fresh heap type. The
    // generated AST types report `__module__ == "ast"` (astcompiler/ast.py:150;
    // the host `_ast.Module.__module__` is likewise `'ast'`).
    let make = |name: &str, base: PyObjectRef| -> PyObjectRef {
        let dict = pyre_object::w_dict_new();
        crate::baseobjspace::setitem(dict, pyre_object::w_str_new("__module__"), pyre_object::w_str_new("ast"))
            .expect("set __module__ on _ast type namespace");
        let args = [
            pyre_object::w_str_new(name),
            pyre_object::w_tuple_new(vec![base]),
            dict,
        ];
        crate::builtins::type_descr_new(&args).expect("_ast heap type creation")
    };

    // Root: AST(object).
    let ast = make("AST", crate::typedef::w_object());
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
