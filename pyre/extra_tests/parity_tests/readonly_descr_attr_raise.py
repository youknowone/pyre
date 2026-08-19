# CPython-suite gap: the descriptor tests assign through a read-only data
# descriptor once, never from a store site hot enough for a JIT raise fold to
# take over.
# parity-tests reason: pyre folds this AttributeError into trace IR instead of
# letting the residual `setattr` build it, so the message, `args`, the
# `__context__` chain and the traceback node are produced by emitted stores.

# `objspace.py:723-740` reaches the descriptor terminal when the receiver keeps
# the default `__setattr__`, the class MRO resolves the name, and the
# descriptor's type resolves no `__set__` but does resolve `__delete__`.  The
# rendered message names the DESCRIPTOR's type, and `__name__` can be
# reassigned without touching that type's version tag, so a fold has to shadow
# the name slot separately from the tag.
#
# No message text is asserted here — it differs per runtime.  What every
# runtime must agree with is ITSELF: `hot_assign`, which runs often enough to
# compile, and `cold_assign`, a separate code object that never does, have to
# answer the same string and the same `args`.

N = 3000


class DeleteOnly:
    def __delete__(self, obj):
        pass


class Holder:
    d = DeleteOnly()


def hot_assign(obj):
    obj.d = 1


def cold_assign(obj):
    obj.d = 1


def cold_exception():
    try:
        cold_assign(Holder())
    except AttributeError as exc:
        return exc
    raise AssertionError("expected AttributeError")


holder = Holder()
expected = cold_exception()

caught = 0
last = None
for _ in range(N):
    try:
        hot_assign(holder)
    except AttributeError as exc:
        caught += 1
        last = exc
assert caught == N, caught
assert type(last) is AttributeError, type(last)
assert str(last) == str(expected), (str(last), str(expected))
assert last.args == expected.args, (last.args, expected.args)
assert last.__traceback__ is not None
assert last.__cause__ is None
assert last.__suppress_context__ is False

# The store must not have happened.
assert "d" not in holder.__dict__, holder.__dict__
assert type(Holder.__dict__["d"]) is DeleteOnly

# Renaming the descriptor's type changes the rendered message without changing
# its version tag.  A fold that pinned only the tag keeps emitting the
# recording-time name, and the cold twin — which reads the live name — then
# disagrees.
DeleteOnly.__name__ = "RenamedDeleteOnly"
renamed = cold_exception()

for _ in range(N):
    try:
        hot_assign(holder)
    except AttributeError as exc:
        last = exc
assert str(last) == str(renamed), (str(last), str(renamed))

# A raise inside an active handler chains `__context__` onto the new instance.
for _ in range(N):
    try:
        raise ValueError("outer")
    except ValueError as outer:
        try:
            hot_assign(holder)
        except AttributeError as inner:
            assert inner.__context__ is outer

# Giving the descriptor's type a `__set__` retires the terminal: the assignment
# now succeeds, and any compiled trace has to side-exit on the version tag.
DeleteOnly.__set__ = lambda self, obj, value: None
for _ in range(N):
    hot_assign(holder)
assert "d" not in holder.__dict__, holder.__dict__

print("OK")
