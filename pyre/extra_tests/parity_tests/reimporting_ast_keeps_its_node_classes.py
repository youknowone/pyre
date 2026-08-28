# CPython-suite gap: `test_ast` builds trees and compares them and
# `test_importlib` reloads modules, but nothing drops `_ast` from `sys.modules`
# and then asks whether a tree it already built is still an `_ast.AST`.
#
# parity-tests reason: the node types are interpreter state rather than module
# state -- `ast.get(space)` / `PyInterpreterState.ast` build them once -- so a
# second `import _ast` publishes the classes the first one did instead of a
# second set.  A runtime that mints them per module dictionary answers
# `isinstance(tree, ast.AST)` with False for every tree built before the
# re-import, and `compile()` reaches `_ast` through `sys.modules` on each call,
# so the two sides stop agreeing about what a tree is.
import ast
import sys

import _ast

first_module = _ast
first_ast = _ast.AST
first_expression = _ast.Expression
tree = compile("x", "<parity>", "eval", ast.PyCF_ONLY_AST)
assert isinstance(tree, first_ast)

del sys.modules["_ast"]
import _ast as again

assert again.AST is first_ast, (again.AST, first_ast)
assert again.Expression is first_expression, (again.Expression, first_expression)
assert isinstance(tree, again.AST), "an existing tree stopped being an AST"
# The same statement from the other side: a tree the re-imported module's
# `compile` builds is an instance of what the first one published.
assert isinstance(compile("x", "<parity>", "eval", ast.PyCF_ONLY_AST), first_ast)

print("OK")
