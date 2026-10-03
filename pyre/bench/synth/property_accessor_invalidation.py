# pyre-check: no-cpython
# `descriptor.py:175 W_Property._immutable_fields_ = ["w_fget?", "w_fset?",
# "w_fdel?"]`.  The `?` is what lets a tracer bake the accessor and equally what
# registers the invalidation an assignment to the slot owes, so re-initialising
# an installed property revokes every loop that folded it.
#
# CPython is not an oracle for this: its `LOAD_ATTR_PROPERTY` specialization
# caches `fget` under the receiver type's version alone, and `property.__init__`
# on an installed descriptor bumps no type's version, so a specialized read
# keeps answering with the previous getter.  Cold, CPython sees the new one —
# the divergence is the specialization's, and pyre follows pypy's `?` instead.
#
# Each rebind happens INSIDE its loop: a read after the loop is interpreted and
# would not consult what the trace baked.  The accessor bodies are residual-free
# so the folds stand rather than aborting.
N = 400000
SWITCH = N // 2


def first_getter(self):
    return 1


def second_getter(self):
    return 2


def first_setter(self, value):
    self.slot = 1


def second_setter(self, value):
    self.slot = 2


class Getter:
    x = property(first_getter)


class Setter:
    slot = 0
    y = property(None, first_setter)


def rebind_getter():
    obj = Getter()
    descr = Getter.__dict__['x']
    total = 0
    i = 0
    while i < N:
        total += obj.x
        if i == SWITCH:
            descr.__init__(second_getter)
        i += 1
    # SWITCH+1 reads of 1, then N-SWITCH-1 reads of 2.
    print('getter', total)


def rebind_setter():
    obj = Setter()
    descr = Setter.__dict__['y']
    total = 0
    i = 0
    while i < N:
        obj.y = i
        total += obj.slot
        if i == SWITCH:
            descr.__init__(None, second_setter)
        i += 1
    print('setter', total)


def drop_getter():
    # The sharper case: the re-init leaves no getter at all, and `W_Property.get`
    # (descriptor.py:224-225) raises rather than calling the old function.
    obj = Getter()
    descr = Getter.__dict__['x']
    raised = 0
    i = 0
    while i < N:
        try:
            obj.x
        except AttributeError:
            raised += 1
        if i == SWITCH:
            descr.__init__(None)
        i += 1
    print('dropped', raised)


rebind_getter()
rebind_setter()
drop_getter()
