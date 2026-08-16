# pyre-check: gate=1

closed = False
async def agen():
    global closed
    try:
        value = yield 1
        yield value
    finally:
        closed = True

g = agen()
try:
    g.__anext__().send(None)
except StopIteration as e:
    first = e.value
try:
    g.asend(42).send(None)
except StopIteration as e:
    second = e.value
try:
    g.aclose().send(None)
except StopIteration:
    close_done = True

assert first == 1
assert second == 42
assert closed
assert close_done
