# pyre-check: gate=1
def f(a, b):
    c = a + b
    return locals()['a'] + locals()['b'] + locals()['c']
result = f(2, 3)

assert result == 10
