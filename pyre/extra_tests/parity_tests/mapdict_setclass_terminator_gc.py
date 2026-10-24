# CPython-suite gap: test_gc never assigns __class__ on a builtin subclass,
# drops every other reference to the new class, and then collects.
# parity-tests reason: after BaseUserClassMapdict.setclass the instance
# class edge is map.terminator.w_cls (Terminator._immutable_fields_);
# native user tracers must visit it. MiniMark traces that field; pyre
# internodes maps as malloc_raw so the collector only sees it through
# the instance custom trace (instance_walk_boxed_storage). The header
# w_class word is gettypefor(typeptr), so a weakref to the previous
# class can clear.
# parity-env: PYPY_GC_NURSERY=4096
# parity-env: MAJIT_GC_NURSERY_POISON=1

"""setclass class edge survives collection of every other reference to C."""

import gc
import weakref


def check(label, base, make, method):
    class B(base):
        pass

    class C(base):
        def marker(self):
            return label

    x = make(B)
    x.__class__ = C
    name = C.__name__
    del C
    del B
    for _ in range(8):
        gc.collect()
    assert type(x).__name__ == name, (label, type(x), type(x).__name__)
    assert method(x), label
    assert x.marker() == label, label
    return x


kept = [
    check("bytes", bytes, lambda B: B(b"ab"), lambda x: x.decode("ascii") == "ab"),
    check("tuple", tuple, lambda B: B((1, 2)), lambda x: x[0] == 1 and len(x) == 2),
    check("set", set, lambda B: B((1, 2)), lambda x: 1 in x),
    check("exc", Exception, lambda B: B("e"), lambda x: x.args == ("e",)),
]
assert len(kept) == 4


def check_old_class_clears(label, base, make):
    class B(base):
        pass

    class C(base):
        pass

    x = make(B)
    wr = weakref.ref(B)
    x.__class__ = C
    del B
    for _ in range(8):
        gc.collect()
    assert wr() is None, (label, wr(), type(x))
    assert type(x).__name__ == "C", (label, type(x).__name__)
    return x


cleared = [
    check_old_class_clears("bytes", bytes, lambda B: B(b"ab")),
    check_old_class_clears("tuple", tuple, lambda B: B((1, 2))),
    check_old_class_clears("set", set, lambda B: B((1, 2))),
    check_old_class_clears("exc", Exception, lambda B: B("e")),
    check_old_class_clears("object", object, lambda B: B()),
]
assert len(cleared) == 5
print("OK")
