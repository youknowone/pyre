# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=inner
# A resumed frame carries a payload — the value send() delivered, the exception
# throw() raised, or a suspended `yield from` delegate — and the JIT entry has
# five paths that decline the frame and hand it back to the plain interpreter.
# A decline that forgets the payload loses a sent value or swallows a thrown
# exception, which is a wrong answer rather than a slow one, so drive every
# resume shape through hot frames and check what came back.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 3


def echo():
    seen = 0
    while True:
        try:
            sent = yield seen
        except ValueError as exc:
            seen += len(exc.args[0])
            continue
        except KeyError as exc:
            seen += len(exc.args[0])
            sent = "after-key"
        seen += len(sent)


def inner(m):
    total = 0
    j = 0
    while j < m:
        total += j
        j += 1
    sent = yield total
    yield len(sent)


def outer(m):
    yield from inner(m)
    yield -1


def terminal(n):
    total = 0
    for i in range(n):
        total += i
    if n < 0:
        yield None
    return total


async def async_terminal(n):
    total = 0
    for i in range(n):
        total += i
    return total


def returned(resumable):
    try:
        resumable.send(None)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("terminal value was yielded")


def drive_send():
    g = echo()
    out = [g.send(None)]
    for value in ("a", "bb", "ccc"):
        out.append(g.send(value))
    return out


def drive_throw():
    g = echo()
    g.send(None)
    out = [g.throw(ValueError("vv"))]
    out.append(g.send("x"))
    out.append(g.throw(KeyError("kkkk")))
    out.append(g.send("y"))
    return out


def drive_yield_from(m):
    g = outer(m)
    out = [g.send(None)]
    out.append(g.send("delegated"))
    return out


def drive_close():
    g = echo()
    g.send(None)
    g.send("z")
    g.close()
    try:
        g.send("w")
    except StopIteration:
        return 1
    return 0


def run(n):
    sent = throwed = delegated = closed = None
    i = 0
    while i < n:
        sent = drive_send()
        throwed = drive_throw()
        delegated = drive_yield_from(100)
        closed = drive_close()
        i += 1
    return sent, throwed, delegated, closed


sent, throwed, delegated, closed = run(N)
print("send:", sent)
print("throw:", throwed)
print("yield-from:", delegated)
print("close:", closed)
assert sent == [0, 1, 3, 6], sent
assert throwed == [2, 3, 16, 17], throwed
assert delegated == [4950, 9], delegated
assert closed == 1, closed
expected = 19_999_900_000
assert returned(terminal(200_000)) == expected
assert returned(async_terminal(200_000)) == expected
print("PASS")
