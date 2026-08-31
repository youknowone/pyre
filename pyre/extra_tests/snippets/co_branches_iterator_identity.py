# pyre-check: gate=1
# CPython-suite gap: `test_monitoring` and `test_code` consume `co_branches()`
# values but do not pin its lazy, distinct iterator object.

def branchy(value):
    for i in range(3):
        if i:
            value += i
        elif value:
            value -= 1
    while value > 0:
        value -= 1
    return value


code = branchy.__code__
iterator = code.co_branches()
assert type(iterator).__name__ == "line_iterator"
assert type(iterator) is not type(code.co_lines())
assert iter(iterator) is iterator
assert list(iterator) == [(24, 28, 112), (40, 46, 66), (76, 82, 86), (120, 126, 148)]
assert list(iterator) == []

first = code.co_branches()
assert next(first) == next(iter(code.co_branches()))
try:
    type(first)()
except TypeError:
    pass
else:
    raise AssertionError("branches iterator type became directly constructible")


class Sub(type(first)):
    pass


assert Sub.__mro__[1] is type(first)
try:
    type(code.co_lines()).__next__(code.co_branches())
except TypeError:
    pass
else:
    raise AssertionError("co_lines iterator accepted a co_branches receiver")

namespace = {}
source = "async def drain(xs):\n    async for item in xs:\n        pass\n"
exec(compile(source, "<branches>", "exec"), namespace)
assert list(namespace["drain"].__code__.co_branches()) == [(24, 28, 40)]
assert list((lambda: 1).__code__.co_branches()) == []
