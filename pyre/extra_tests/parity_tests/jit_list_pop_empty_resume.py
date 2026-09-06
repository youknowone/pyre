# CPython-suite gap: test_pickle does not deterministically compile the
# non-empty list.pop path before an underflow and traceback frame.clear().
# parity-tests reason: exercise generated pop guards and exception resume
# with both Object and Integer list storage.

import pickle
import traceback

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass


class Reader:
    pass


def pop_twice(items):
    first = items.pop()
    second = items.pop()
    return first + second


def run():
    reader = Reader()
    load_setitem = pickle._Unpickler.load_setitem
    for _ in range(1000):
        reader.stack = [{}, 4, None]
        load_setitem(reader)
        assert reader.stack == [{4: None}]
        assert pop_twice([3, 7]) == 10

    for _ in range(100):
        for stack in ([], [None], [None, None]):
            reader.stack = stack
            try:
                load_setitem(reader)
            except IndexError as exc:
                assert reader.stack == []
                traceback.clear_frames(exc.__traceback__)
            else:
                raise AssertionError("expected unpickler stack underflow")

        items = [7]
        try:
            pop_twice(items)
        except IndexError as exc:
            assert str(exc) == "pop from empty list"
            assert items == []
            traceback.clear_frames(exc.__traceback__)
        else:
            raise AssertionError("expected integer list underflow")


run()
print("OK")
