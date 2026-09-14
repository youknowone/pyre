# pyre-check: max-pypy-ratio=83
# pyre-check: max-wasm-ratio=6.0
# `_opimpl_recursive_call` looks inside on bridges of a compiled portal
# (`pyjitpl.py`) until `max_unroll_recursion`. wasm compiles each extra
# trace as its own module; ubuntu-24.04 run 34810630624 measured 5.2x
# against dynasm, so 6.0x is that reading plus WASM_RATIO_FIT_HEADROOM.
# PEP 709 inlines comprehensions into their enclosing scope. At module
# scope the iteration variable becomes a CO_FAST_HIDDEN local, yet a later
# top-level binding of the same name is a normal global (STORE_NAME). The
# fast<->locals sync must skip the hidden slot, otherwise the next global
# store erases the just-bound name and reading it raises NameError.

squares = [n for n in range(5)]   # noqa: C416 - `n` is a module-scope hidden fast local (comprehension form is the point)
n = 100                           # rebind `n` as a real global
total = 0                         # a second global store must not erase `n`

i = 0
while i < 300000:
    total = total + n
    i = i + 1

print(squares[-1], n, total)
