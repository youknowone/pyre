"""Python 3.14 coroutine-origin tracking accepts ``depth`` by keyword."""

import sys


old_depth = sys.get_coroutine_origin_tracking_depth()
try:
    assert sys.set_coroutine_origin_tracking_depth(depth=1) is None
    assert sys.get_coroutine_origin_tracking_depth() == 1

    try:
        sys.set_coroutine_origin_tracking_depth(1, depth=2)
    except TypeError:
        pass
    else:
        raise AssertionError("depth accepted both positionally and by keyword")

    try:
        sys.set_coroutine_origin_tracking_depth(unknown=1)
    except TypeError:
        pass
    else:
        raise AssertionError("unknown keyword was accepted")
finally:
    sys.set_coroutine_origin_tracking_depth(old_depth)

print("OK")
