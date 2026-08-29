"""A suspended generator collected by the GC runs its finally / with cleanup.

generator.py `_finalize_`: when a generator that is still suspended inside a
try/finally, except, or with block is garbage-collected, a GeneratorExit is
raised into it so the pending cleanup runs.
"""

import gc

# try/finally
log = []


def g_finally():
    try:
        yield 1
    finally:
        log.append("finally")


gg = g_finally()
next(gg)
del gg
gc.collect()
assert log == ["finally"], log

# with-statement cleanup (context manager __exit__)
log = []


class CM:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        log.append("exit")
        return False


def g_with():
    with CM():
        yield 1


gg = g_with()
next(gg)
del gg
gc.collect()
assert log == ["exit"], log

# nested try/finally — both handlers run, inner before outer
log = []


def g_nested():
    try:
        try:
            yield 1
        finally:
            log.append("inner")
    finally:
        log.append("outer")


gg = g_nested()
next(gg)
del gg
gc.collect()
assert log == ["inner", "outer"], log

# yield-from: the delegate's finally runs
log = []


def sub():
    try:
        yield 1
    finally:
        log.append("sub")


def g_yieldfrom():
    yield from sub()


gg = g_yieldfrom()
next(gg)
del gg
gc.collect()
assert log == ["sub"], log

# A finalizer closing one yield-from chain must not recursively start the
# queue drain for every other unreachable chain.  pathlib/glob keeps many of
# these suspended at once while traversing deep wildcard patterns.
log = []


def chain_leaf(tag):
    try:
        yield tag
    finally:
        log.append((tag, 0))


def chain_wrap(delegate, tag, depth):
    try:
        yield from delegate
    finally:
        log.append((tag, depth))


chains = []
chain_count = 40
chain_depth = 30
for tag in range(chain_count):
    chain = chain_leaf(tag)
    for depth in range(1, chain_depth + 1):
        chain = chain_wrap(chain, tag, depth)
    next(chain)
    chains.append(chain)
del chain
del chains
gc.collect()
assert len(log) == chain_count * (chain_depth + 1), len(log)

# a generator suspended outside any handler needs no cleanup
log = []


def g_plain():
    yield 1
    log.append("unreachable")


gg = g_plain()
next(gg)
del gg
gc.collect()
assert log == [], log

# an exhausted generator does not re-run its finally when later collected
log = []


def g_done():
    try:
        yield 1
    finally:
        log.append("done")


gg = g_done()
list(gg)  # exhaust — finally runs here
assert log == ["done"], log
del gg
gc.collect()
assert log == ["done"], log
