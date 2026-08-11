# pyre-check: no-cpython

import contextvars
import gc


name = "runtime-context-" + ("x" * 29)
variable = contextvars.ContextVar(name)
rendered = repr(variable)

assert "runtime-context-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx" in rendered
assert any(obj is rendered for obj in gc.get_objects())

token = variable.set(object())
token_rendered = repr(token)

assert "runtime-context-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx" in token_rendered
assert any(obj is token_rendered for obj in gc.get_objects())

print("contextvars reprs are collectable")
