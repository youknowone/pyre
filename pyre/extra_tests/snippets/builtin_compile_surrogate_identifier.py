# pyre-check: gate=1
# Every identifier field of an AST object reaches the compiler as a `str`, and
# a `str` may carry a lone surrogate.  The reference accepts one and keeps it
# in `co_names`; pyre's identifiers are UTF-8, so it reports the encoding
# failure instead.  What neither may do is take the process down, which is
# what reading the field through an infallible `&str` accessor did.
import ast

SURROGATE = chr(0xD800)


def outcome(build):
    try:
        compile(build(), "<surrogate>", "exec")
    except UnicodeEncodeError:
        return "encode-error"
    return "compiled"


def alias_name():
    tree = ast.parse("import os")
    tree.body[0].names[0].name = SURROGATE
    return tree


def global_names():
    tree = ast.parse("global x")
    tree.body[0].names[0] = SURROGATE
    return tree


def import_from_module():
    tree = ast.parse("from a import b")
    tree.body[0].module = SURROGATE
    return tree


def keyword_arg():
    tree = ast.parse("f(a=1)")
    tree.body[0].value.keywords[0].arg = SURROGATE
    return tree


def constant_kind():
    tree = ast.parse("x = 'a'")
    tree.body[0].value.kind = SURROGATE
    return tree


def argument_name():
    tree = ast.parse("def f(a): pass")
    tree.body[0].args.args[0].arg = SURROGATE
    return tree


def type_param_name():
    tree = ast.parse("def f[T](): pass")
    tree.body[0].type_params[0].name = SURROGATE
    return tree


def except_handler_name():
    tree = ast.parse("try:\n    pass\nexcept E as e:\n    pass\n")
    tree.body[0].handlers[0].name = SURROGATE
    return tree


def match_capture_name():
    tree = ast.parse("match v:\n    case [x]:\n        pass\n")
    tree.body[0].cases[0].pattern.patterns[0].name = SURROGATE
    return tree


def type_comment():
    tree = ast.parse("x = 1  # type: int", type_comments=True)
    tree.body[0].type_comment = SURROGATE
    return tree


for build in (
    alias_name,
    global_names,
    import_from_module,
    keyword_arg,
    constant_kind,
    argument_name,
    type_param_name,
    except_handler_name,
    match_capture_name,
    type_comment,
):
    assert outcome(build) in ("compiled", "encode-error"), build.__name__
