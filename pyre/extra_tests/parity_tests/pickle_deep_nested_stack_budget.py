"""Legal pickle recursion must survive an ordinary surrounding call stack."""

import io
import pickle


data = None
for _ in range(100):
    data = [data]


def dump_below_outer_frames(depth):
    if depth:
        return dump_below_outer_frames(depth - 1)
    buffer = io.BytesIO()
    pickle._Pickler(buffer, protocol=4).dump(data)
    return buffer.getvalue()


payload = dump_below_outer_frames(20)
assert pickle._Unpickler(io.BytesIO(payload)).load() == data
print("OK")
