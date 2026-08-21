# pyre-check: gate=1
"""A traced frame hands back its return value at the value's live address.

`sys.settrace`'s `return` callback and `sys.setprofile`'s `leaveframe`
callback are application-level Python, so a freshly allocated return value
moves under them.  Both hooks were handed the value by copy and the caller
was given back the word taken before the callback ran.

Three things are load-bearing:
  * the value must be freshly allocated and MOVABLE -- returning an argument
    or an int leaves nothing for a minor collection to relocate;
  * the caller must KEEP it, since a discarded result is never dereferenced;
  * the callback must allocate and collect, which is what moves it.
"""

import gc
import sys

KEEP = None
TICKS = 0


def churn():
    global KEEP
    KEEP = [[i] * 12 for i in range(20)] + [bytearray(b"Q" * 64) for _ in range(10)]
    gc.collect()
    KEEP = [[i] * 12 for i in range(20)] + [bytearray(b"Q" * 64) for _ in range(10)]


def tracer(frame, event, arg):
    global TICKS
    TICKS += 1
    if TICKS % 7 == 0:
        churn()
    return tracer


def make_list(n):
    return [n]


def make_dict(n):
    return {n: n}


held = []
sys.settrace(tracer)
for i in range(40):
    held.append(make_list(i))
    held.append(make_dict(i))
sys.settrace(None)

sys.setprofile(tracer)
for i in range(40):
    held.append(make_list(i))
sys.setprofile(None)

churn()
for value in held:
    if type(value) is list:
        value.append(0)
    else:
        value["z"] = 0
churn()
print(len(held), sorted({type(v).__name__ for v in held}))
print(held[0], held[1], held[-1])
