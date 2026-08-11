# pyre-check: no-cpython

import contextvars
import gc


name = "runtime-context-" + ("x" * 29)
variable = contextvars.ContextVar(name)
rendered = repr(variable)

assert "runtime-context-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx" in rendered
assert any(obj is rendered for obj in gc.get_objects())

print("ContextVar repr is collectable")
