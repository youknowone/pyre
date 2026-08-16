# pyre-check: gate=1
x = 7
def f(a, *, b=3):
    return a + b + x
g = f.__globals__
code = f.__code__

assert g['x'] == 7
assert code.co_argcount == 1
assert code.co_kwonlyargcount == 1
assert code.co_name == 'f'
assert code.co_varnames[0] == 'a'
