# pyre-check: no-cpython

import gc
import types


# Exercise the explicit descriptor path.  repr(alias) uses the interpreter's
# generic display helper, while GenericAlias.__repr__(alias) calls ga_repr.
name = "RuntimeAlias" + ("X" * 19)
runtime_type = type(name, (), {})
alias = list[runtime_type]
rendered = types.GenericAlias.__repr__(alias)

assert rendered == "list[__main__.RuntimeAliasXXXXXXXXXXXXXXXXXXX]"
assert any(obj is rendered for obj in gc.get_objects())

print("generic alias repr is collectable")
