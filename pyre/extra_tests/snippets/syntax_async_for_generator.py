# pyre-check: gate=1

async def agen():
    yield 2
    yield 3

preserved_exception = False
async def consume():
    global preserved_exception
    result = []
    try:
        raise ValueError('outer')
    except ValueError:
        async for value in agen():
            result.append(value)
        try:
            raise
        except ValueError:
            preserved_exception = True
    return result

coro = consume()
try:
    coro.send(None)
except StopIteration as e:
    result = e.value

assert result == [2, 3]
assert preserved_exception
