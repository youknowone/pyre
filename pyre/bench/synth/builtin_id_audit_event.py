# `id()` emits the `builtins.id` audit event carrying the value it is about to
# return (`operation.py:84-88`: `w_res = space.id(w_object)`, then
# `space.audit("builtins.id", [w_res])`).  pyre computed the id and returned it
# without ever emitting, so `sys.addaudithook` could not observe the call.
#
# The hook records a count and the last value, so two things are checked at
# once: one event per call, and the event's argument being what the call handed
# back.  Nothing in the hook body calls `id()`, so the count cannot feed itself.

import sys

count = [0]
last_seen = [0]


def hook(event, args):
    if event == "builtins.id":
        count[0] += 1
        last_seen[0] = args[0]


obj = [1, 2, 3]
sys.addaudithook(hook)

N = 2000
before = count[0]
last_returned = 0
for _ in range(N):
    last_returned = id(obj)
emitted = count[0] - before

assert emitted == N, "%d events for %d calls" % (emitted, N)
assert last_seen[0] == last_returned, "event carried %r, call returned %r" % (
    last_seen[0],
    last_returned,
)

print("OK")
