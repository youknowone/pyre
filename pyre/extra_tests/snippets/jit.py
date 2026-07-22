def foo():
    a = 5
    return 10 + a


def bar():
    a = 1e6
    return a / 5.0


def baz(a: int, b: int):
    return a + b + 12


def many_args(*args):
    return len(args)


def two_unsupported_calls():
    # The JIT helper ABI intentionally tops out at 14 explicit arguments.
    # Two consecutive larger calls must produce one clean interpreter bailout,
    # not two exits attached to the same flow-graph block.
    first = many_args(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    second = many_args(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    return first + second


def tests():
    assert foo() == 15
    assert bar() == 2e5
    assert baz(17, 20) == 49
    assert baz(17, 22.5) == 51.5
    assert two_unsupported_calls() == 30


tests()

if hasattr(foo, "__jit__"):
    print("Has jit")
    foo.__jit__()
    bar.__jit__()
    baz.__jit__()
    two_unsupported_calls.__jit__()
    tests()
