# pyre-check: gate=1
# pyre-check: pypy-diverges: pins `__match_args__` on every `_ast` type, the base included; pypy3 binds it on the concrete nodes but `ast.AST` carries none
# CPython-suite gap: `test_ast` reads `_fields` on every node type but never
# reads `__match_args__`, and its pattern-matching tests build their own
# classes.  A runtime that binds only `_fields` passes the whole module while
# every `case ast.Expr(value)` in the standard library silently stops matching.
#
# parity-tests reason: `make_type` builds ONE tuple of field names and binds it
# to both names, so the two are the same object on every `_ast` type, field-less
# ones included.  The observable consequence is not the attribute itself:
# `traceback._extract_caret_anchors_from_line_segment` is written as class
# patterns over `ast.BinOp` / `ast.Subscript` / `ast.Call`, so without
# `__match_args__` a class pattern with positional sub-patterns raises
# `TypeError: Expr() accepts 0 positional sub-patterns` and every traceback
# rendered through `traceback.py` loses its `~`/`^` anchor row.
#
# PyPy binds `__match_args__` the same way, `Pass` and its empty tuple
# included, but only in `State.make_new_type`, which runs over `AST_TYPES`.
# The base is `space.gettypeobject(W_AST.typedef)` and never goes through it,
# and that typedef lists `_fields` and `_attributes` alone -- so `ast.AST`
# carries no `__match_args__` and the last line of the surface check cannot
# hold there.  Structural, not a version gap: unchanged from 7.3.20 through
# the 7.3.24 checkout under this repository.

import ast
import sys
import traceback

print("_fields is __match_args__:", ast.Expr._fields is ast.Expr.__match_args__)
print("Expr.__match_args__:", ast.Expr.__match_args__)
# A node type with no fields still carries the (empty) tuple, as does the base.
print("Pass/AST:", ast.Pass.__match_args__, ast.AST.__match_args__)

TREE = ast.parse("a + b")
match TREE.body[0]:
    case ast.Expr(ast.BinOp(left, op, right)):
        print("matched:", type(left).__name__, type(op).__name__, type(right).__name__)
    case _:
        print("class pattern did not match")


def boom(mapping):
    return mapping["a"] + mapping["b"]


try:
    boom({"a": 1})
except KeyError:
    report = "".join(traceback.format_exception(*sys.exc_info()))
    for line in report.splitlines():
        # Only the anchor rows and the frame bodies are compared; the file
        # path and line numbers move with this file.
        stripped = line.strip()
        if stripped.startswith("File "):
            continue
        print(repr(stripped))

print("OK")
