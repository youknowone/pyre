# pyre-check: gate=1
# CPython-suite gap: test_ast and test_symtable use these constants through the
# module that publishes them, so both sides of a comparison move together and
# a wrong number is invisible.
# parity-tests reason: `ast.PyCF_TYPE_COMMENTS` carried `0x40000000` and
# `_symtable.DEF_BOUND` carried an extra `DEF_TYPE_PARAM` bit; both are
# documented numbers a caller can spell for itself.

# pyre-check: pypy-diverges: pypy3 has no `_symtable` at all, and `consts.py`
# keeps `PyCF_ASYNC_HACKS` on `0x1000` so it moves `PyCF_TYPE_COMMENTS` out to
# `0x40000000`; 3.14 has no `PyCF_ASYNC_HACKS`.

import ast
import symtable
import _symtable

# `Include/cpython/compile.h`.  The whole `PyCF_*` set is bits 0x0100..0x10000.
assert ast.PyCF_ONLY_AST == 0x0400, hex(ast.PyCF_ONLY_AST)
assert ast.PyCF_TYPE_COMMENTS == 0x1000, hex(ast.PyCF_TYPE_COMMENTS)
assert ast.PyCF_ALLOW_TOP_LEVEL_AWAIT == 0x2000, hex(ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
assert ast.PyCF_OPTIMIZED_AST == 0x8000 | ast.PyCF_ONLY_AST, hex(ast.PyCF_OPTIMIZED_AST)

# The number has to keep working as the flag it names: `compile()` takes it,
# and rejects the one it used to be.  What the flag then *collects* is a
# separate question, and `ast_type_comments.py` is where it is asserted.
tree = compile(
    "x = 1  # type: int\n",
    "<s>",
    "exec",
    flags=ast.PyCF_ONLY_AST | ast.PyCF_TYPE_COMMENTS,
)
assert isinstance(tree, ast.Module), tree

try:
    compile("x = 1\n", "<s>", "exec", flags=ast.PyCF_ONLY_AST | 0x4000_0000)
except ValueError as e:
    assert str(e) == "compile(): unrecognised flags", e
else:
    raise AssertionError("0x40000000 is not a compile flag")

# `pycore_symtable.h`: DEF_BOUND is DEF_LOCAL | DEF_PARAM | DEF_IMPORT.
assert _symtable.DEF_LOCAL == 2 << 0
assert _symtable.DEF_PARAM == 2 << 1
assert _symtable.DEF_IMPORT == 2 << 6
assert _symtable.DEF_TYPE_PARAM == 2 << 9
assert _symtable.DEF_BOUND == 134, _symtable.DEF_BOUND
assert _symtable.DEF_BOUND == (
    _symtable.DEF_LOCAL | _symtable.DEF_PARAM | _symtable.DEF_IMPORT
)

# `symtable.py` reads DEF_BOUND in `is_global` and `is_local`, and in both only
# for a symbol in the module block.  DEF_TYPE_PARAM is set only on names inside
# a `type parameters` block, so the bit that was dropped had no reader.
top = symtable.symtable(
    "type Alias[T] = list[T]\ndef f[U](x: U) -> U:\n    return x\n", "<s>", "exec"
)
assert top.get_type() == "module", top.get_type()
assert [s.get_name() for s in top.get_symbols() if s.is_type_parameter()] == []
blocks = {c.get_name(): c for c in top.get_children()}
for owner, param in (("Alias", "T"), ("f", "U")):
    block = blocks[owner]
    assert block.get_type() == "type parameters", (owner, block.get_type())
    sym = block.lookup(param)
    assert sym.is_type_parameter(), owner
    # It carries DEF_LOCAL as well, and its block is not the module block, so
    # both readers answer from the scope rather than from the mask.
    assert sym.is_assigned(), owner
    assert sym.is_local() and not sym.is_global(), owner

print("OK")
