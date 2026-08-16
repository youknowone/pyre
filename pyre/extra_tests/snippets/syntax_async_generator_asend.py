# pyre-check: gate=1

async def agen():
    yield 42

g = agen()
frame_visible = g.ag_frame is not None
code_visible = g.ag_code is not None
initially_suspended = g.ag_suspended
a = g.__anext__()
try:
    a.send(None)
except StopIteration as e:
    yielded = e.value
try:
    g.__anext__().send(None)
except StopAsyncIteration:
    exhausted = True

assert yielded == 42
assert exhausted
assert frame_visible
assert code_visible
assert not initially_suspended
