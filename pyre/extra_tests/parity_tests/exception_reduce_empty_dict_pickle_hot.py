# CPython-suite gap: `test.test_pickle` dumps exceptions. Accessing
# `__dict__` first sets the instance dict, so `__reduce__` packs three
# items and the walker records that packing
# (`try_walker_specialize_exception_reduce`).
#
# parity-tests reason: a residual `bh_call_fn(__reduce__)` during that
# walk collects while `WalkFrameState` is borrowed
# (`walk_frame_state_roots`).
import pickle


def dump_with_empty_dict(exc):
    exc.__dict__
    got = pickle.loads(pickle.dumps(exc))
    assert got.args == exc.args
    return got


for i in range(8000):
    dump_with_empty_dict(ValueError(i))

e = ValueError("m")
e.__dict__
assert pickle.loads(pickle.dumps(e)).args == ("m",)

print("OK")
