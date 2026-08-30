# pyre-check: gate=1
"""`del sys.modules['_ast']; import _ast` republishes the generated hierarchy.

The node types are built once and the `Load` singleton is pinned to that first
build, so a second initializer run must publish the same objects rather than
mint a parallel set: `ast_init` fills the `ctx` default by identity, and a
second `expr_context` names a type that test never matches.

`moduledef.py` resolves every name through `ast.get(space).w_<Name>`, so what a
later import publishes comes from the hierarchy itself.  Rebinding a name on
the module the initializer first filled must therefore stay invisible to it.
"""

import ast
import sys


def reimport():
    del sys.modules["_ast"]
    import _ast

    return _ast


first = __import__("_ast")
first_name = first.Name
first_only_ast = first.PyCF_ONLY_AST

# The `ctx` default is filled by identity against the pinned `Load`.
assert ast.parse("x").body[0].value.ctx.__class__ is ast.Load

second = reimport()
assert second.Name is first_name, (second.Name, first_name)
assert second.PyCF_ONLY_AST == first_only_ast
assert ast.parse("x").body[0].value.ctx.__class__ is ast.Load
assert isinstance(ast.Name(id="x").ctx, ast.Load)

# `first` is the namespace the initializer filled.  A rebinding there belongs
# to that module object, not to the hierarchy a later import reads.
first.Name = 42
assert first.Name == 42
third = reimport()
assert third.Name is first_name, (third.Name, first_name)
assert ast.parse("x").body[0].value.ctx.__class__ is ast.Load
assert isinstance(ast.Name(id="x").ctx, ast.Load)
