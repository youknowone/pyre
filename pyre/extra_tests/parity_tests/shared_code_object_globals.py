# CPython-suite gap: exec tests reuse a code object across namespaces but never
# make either copy hot enough to be compiled.
# parity-tests reason: this guards pyre's LOAD_GLOBAL module-cell fold, which
# has a namespace to pin only because the JIT records one.

"""One code object, two namespaces, both hot.

`pycode.py frame_stores_global` publishes the FIRST globals dictionary on the
code object; a frame whose globals differ carries the override in its own
`debugdata` (`pyframe.py get_w_globals`).  The warm key is `(code, pc)`, so the
loop compiled for the first namespace is the loop the second namespace's frame
enters, and the module-cell fold behind `LOAD_GLOBAL` has to test the live
frame rather than the namespace it recorded.

Each module reads a different `G`, so a fold bound to the recording namespace
answers with the other module's value (or a mixture, once one iteration deopts).
Both totals are exact multiples of their own `G`, and neither is a multiple of
the other's, so a wrong fold cannot produce a passing number.
"""

SRC = """
def hot():
    t = 0
    for _ in range(60000):
        t += G
    return t
"""

code = compile(SRC, "<shared>", "exec")

import sys

# `type(sys)` rather than `types.ModuleType`: the wasm backend's stdlib does
# not carry `types`, and the class is the same object either way.
module_type = type(sys)
m1 = module_type("shared_globals_m1")
m2 = module_type("shared_globals_m2")
m1.G = 1
m2.G = 2
sys.modules["shared_globals_m1"] = m1
sys.modules["shared_globals_m2"] = m2

exec(code, m1.__dict__)
exec(code, m2.__dict__)

# Interleaved, so the second namespace enters the loop the first one compiled
# and vice versa, and so a deopt in either direction is retried while hot.
for round_ in range(6):
    a = m1.hot()
    b = m2.hot()
    assert a == 60000, f"round {round_}: shared_globals_m1.hot() = {a}"
    assert b == 120000, f"round {round_}: shared_globals_m2.hot() = {b}"

# Rebinding through one namespace must not reach the other.
m1.G = 10
assert m1.hot() == 600000, m1.hot()
assert m2.hot() == 120000, m2.hot()

print("OK")
