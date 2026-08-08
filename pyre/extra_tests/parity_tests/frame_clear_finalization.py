import gc
import sys
import time
import traceback
import weakref


def assert_frame_clear_releases_after_collect():
    def make_frame():
        class Watched:
            pass

        obj = Watched()
        ref = weakref.ref(obj)

        def inner(arg):
            local = arg
            local
            raise RuntimeError

        try:
            inner(obj)
        except RuntimeError as exc:
            frames = []
            tb = exc.__traceback__
            while tb is not None:
                frames.append(tb.tb_frame)
                tb = tb.tb_next
            return frames, ref

    frames, ref = make_frame()
    assert ref() is not None
    for frame in frames:
        frame.clear()
    gc.collect()
    assert ref() is None


def assert_generator_close_suspended_finalizes():
    seen = []

    class Watched:
        def __del__(self):
            seen.append("del")

    def gen():
        obj = Watched()
        yield obj

    generator = gen()
    next(generator)
    generator.close()
    assert seen == ["del"]


def assert_generator_close_unstarted_clears_frame():
    seen = []

    class Watched:
        def __del__(self):
            seen.append("del")

    def make_generator():
        obj = Watched()

        def gen(arg):
            yield arg

        return gen(obj)

    generator = make_generator()
    generator.close()
    assert generator.gi_frame is None
    assert seen in ([], ["del"])
    del generator
    gc.collect()
    assert seen == ["del"]


def assert_generator_close_finally_finalizes():
    seen = []

    class Watched:
        def __del__(self):
            seen.append("del")

    def gen():
        obj = Watched()
        try:
            yield obj
        finally:
            seen.append("finally")

    generator = gen()
    next(generator)
    generator.close()
    assert seen == ["finally", "del"]


def assert_executing_frame_clear_raises():
    try:
        sys._getframe().clear()
    except RuntimeError:
        pass
    else:
        raise AssertionError("frame.clear() on an executing frame must raise")


def assert_assert_raises_shape_stays_cheap():
    class Expected(Exception):
        pass

    class Check:
        def assertRaises(self, exc_type, func):
            try:
                func()
            except exc_type as exc:
                traceback.clear_frames(exc.__traceback__)
            else:
                raise AssertionError("expected exception")

    def raises():
        raise Expected

    check = Check()
    count = 200
    # `process_time` rather than `resource.getrusage`: the CPU this process has
    # spent is the measurement, and `resource` is a POSIX module, so importing
    # it would take the five checks above off Windows along with this one.
    before = time.process_time()
    for _ in range(count):
        check.assertRaises(Expected, raises)
    after = time.process_time()
    per_op_ms = (after - before) * 1000.0 / count
    assert per_op_ms < 1.0, per_op_ms


assert_frame_clear_releases_after_collect()
assert_generator_close_suspended_finalizes()
assert_generator_close_unstarted_clears_frame()
assert_generator_close_finally_finalizes()
assert_executing_frame_clear_raises()
assert_assert_raises_shape_stays_cheap()

print("OK")
