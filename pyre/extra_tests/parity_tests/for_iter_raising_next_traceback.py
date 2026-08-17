# CPython-suite gap: the suite never inspects the traceback of a
# non-StopIteration exception raised out of __next__ by a hot FOR_ITER.
# parity-tests reason: FOR_ITER's mismatch arm re-raises a value that already
# carries this frame's traceback node, so the re-raise must not record a second
# one.

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass

import sys
import traceback


class Boom:
    def __iter__(self):
        return self

    def __next__(self):
        return self.missing


def enclosed():
    for value in Boom():
        return value


def bare():
    for value in Boom():
        return value


shapes = set()
coordinates = set()
for _ in range(8):
    try:
        try:
            enclosed()
        except AttributeError:
            raise
    except AttributeError:
        tb = sys.exc_info()[2]
        shapes.add(tuple(f.name for f in traceback.extract_tb(tb)))
        coordinates.add(tuple((f.f_code.co_name, lasti) for f, lasti in traceback.walk_tb(tb)))

    try:
        bare()
    except AttributeError:
        tb = sys.exc_info()[2]
        shapes.add(tuple(f.name for f in traceback.extract_tb(tb)))
        coordinates.add(tuple((f.f_code.co_name, lasti) for f, lasti in traceback.walk_tb(tb)))

assert shapes == {
    ("<module>", "enclosed", "__next__"),
    ("<module>", "bare", "__next__"),
}, sorted(shapes)
assert coordinates == {
    (("<module>", 338), ("enclosed", 22), ("__next__", 4)),
    (("<module>", 170), ("bare", 22), ("__next__", 4)),
}, sorted(coordinates)
print("OK")
