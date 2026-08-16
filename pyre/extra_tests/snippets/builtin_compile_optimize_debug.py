# pyre-check: gate=1
import _ast
raw = compile('a*__debug__', 'f.py', 'exec', flags=_ast.PyCF_ONLY_AST)
opt_source = compile('a*__debug__', 'f.py', 'exec', flags=_ast.PyCF_OPTIMIZED_AST)
opt_tree = compile(raw, 'f.py', 'exec', flags=_ast.PyCF_OPTIMIZED_AST)
result = (
    isinstance(raw.body[0].value.right, _ast.Name)
    and isinstance(opt_source.body[0].value.right, _ast.Constant)
    and opt_source.body[0].value.right.value is True
    and isinstance(opt_tree.body[0].value.right, _ast.Constant)
    and opt_tree.body[0].value.right.value is True
)

assert result
