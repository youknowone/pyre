# pyre-check: gate=1
"""`ContextVar.set`/`reset`/`get` address their operands at their live places.

Every step of these three is application-level Python: the Context mapping's
`__getitem__`/`set`/`delete`, and the attribute reads around them.  The values
that outlive one of those calls -- the Context, its `_data` mapping, the old
binding, and the caller's own variable and value -- were taken by copy before
it.  The gateway hands a builtin a native argument array it rebuilt from its
own roots, so `args` is not rewritten by a collection either.

`reset` compares addresses to decide whether the token belongs to this
variable and this Context, so an operand that moved there does not merely
crash: it reports a token as belonging to a different ContextVar.

The stored value has to be MOVABLE -- `set(1)` interns an immortal int and
stays clean whatever happens.

The collection point is a `sys.settrace` callback that allocates: these steps
are attribute reads and small mapping calls, so on their own they do not move
the nursery far enough to relocate what the caller is holding.  A tracer makes
every one of them a collection point, which is what the operands have to
survive.
"""

import contextvars
import gc
import sys

KEEP = None
TICKS = 0


def churn():
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]
    gc.collect()
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


def tracer(frame, event, arg):
    global TICKS
    TICKS += 1
    if TICKS % 11 == 0:
        churn()
    return tracer


variable = contextvars.ContextVar("gate")
held = []
sys.settrace(tracer)
for i in range(30):
    churn()
    token = variable.set([i, i + 1])
    held.append(token)
    assert variable.get() == [i, i + 1], variable.get()

churn()
for i, token in enumerate(reversed(held)):
    variable.reset(token)

churn()
missing = contextvars.ContextVar("absent")
for i in range(30):
    churn()
    assert missing.get([i]) == [i], missing.get([i])

sys.settrace(None)
churn()
for token in held:
    value = token.old_value
    if type(value) is list:
        value.append(0)
print(len(held), held[0].old_value, held[-1].old_value)
