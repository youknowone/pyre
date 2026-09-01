# pyre-check: max-pypy-ratio=25
# pyre-check: skip-cpython
# cpython 4.45s vs pyre 0.21s (21.2x on the ubuntu runner), and it is not
# gated on — only pypy is.
# A bare `raise X` of an exception kind whose `descr_init` writes flattened
# slots beyond `args_w` — `interp_exceptions.py:496-499 W_StopIteration`
# (`value`), `:810-812 W_NameError` and `:1134-1137 W_AttributeError` (`name` /
# `obj`), `:363-377 W_ImportError` (`name` / `path` / `msg`). `do_raise`
# instantiates the class with no arguments, so every one of those slots takes a
# trace-time constant and the construction can be traced instead of residualised.
# Each raise site stays monomorphic so the class operand is a single value.
#
# N is large because the residual shape this replaces degrades super-linearly:
# 30k iterations hide it inside compile time, while 3M separate the two shapes
# by two orders of magnitude.
N = 6000000


def stop_iteration():
    raise StopIteration


def name_error():
    raise NameError


def attribute_error():
    raise AttributeError


def import_error():
    raise ImportError


def main():
    acc = 0
    i = 0
    while i < N:
        try:
            stop_iteration()
        except StopIteration:
            acc = acc + 1
        try:
            name_error()
        except NameError:
            acc = acc + 2
        try:
            attribute_error()
        except AttributeError:
            acc = acc + 4
        try:
            import_error()
        except ImportError:
            acc = acc + 8
        i = i + 1
    print(acc)


main()


def caught(fn, exc_type):
    try:
        fn()
    except exc_type as exc:
        return exc
    raise AssertionError("expected exception")


payload = None


def stop_value():
    raise StopIteration(payload)


def import_value():
    raise ImportError(payload)


# Correctness belongs beside the performance census: the emitted constructor
# must initialize every flattened slot, and runtime operands must remain live
# after the trace was recorded with None.
for _ in range(3000):
    assert caught(stop_iteration, StopIteration).value is None
    assert caught(name_error, NameError).name is None
    attr = caught(attribute_error, AttributeError)
    assert (attr.name, attr.obj) == (None, None)
    imp = caught(import_error, ImportError)
    assert (imp.name, imp.path, imp.msg) == (None, None, None)
    assert caught(stop_value, StopIteration).value is None
    assert caught(import_value, ImportError).msg is None

for payload in range(3000):
    stop = caught(stop_value, StopIteration)
    assert stop.value == payload and stop.args == (payload,)
    text = str(payload)
    payload = text
    imp = caught(import_value, ImportError)
    assert imp.msg == text and imp.args == (text,)
