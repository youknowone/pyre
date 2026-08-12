# CPython-suite gap: the suite never runs `isinstance` hot enough to trace over
# an instance whose class binds its own `__class__`.
# parity-tests reason: a traced `isinstance` must keep calling a `__class__`
# property and keep returning what that property implies.

# `isinstance(obj, cls)` falls back to reading `obj.__class__` whenever the MRO
# test misses (`abstractinst.py:76`).  That read goes through the ordinary
# getattr, so a class binding `__class__` as a property has its getter run — by
# the *default* `object.__getattribute__`, which consults the type's data
# descriptors before anything else.  The getter is user code, and it decides
# the answer: `isinstance(Masked(), int)` is True below, not the plain False a
# miss looks like.  Every count is printed rather than asserted, so a fold that
# elided the call, cached its result, or answered False would show up as a diff
# against CPython instead of a silent pass.
#
# Scope, measured rather than assumed: this pins the observable semantics only.
# It does NOT exercise `observed_replay_safe_isinstance` in
# `jitcode_dispatch/residual_call.rs`, whose sole consumer is the nested
# residual abort inside an inline sub-walk; at this call depth the residual is
# a plain `call_may_force` and that gate no-ops.  Swapping that predicate for
# the weaker one it replaced leaves every line here byte-identical.
#
# The gate's own witness is `bench/synth/foriter_isinstance_class_property_replay.py`,
# which needs three things this file has none of: the call admitted into a
# FOR_ITER body, an opaque trailing call to make the first sub-walk abort, and
# a replay to turn the missing effect record into an N+1 hit count.

N = 300

calls = []


class Masked:
    """`__class__` is a property, so the miss path runs Python."""

    @property
    def __class__(self):
        calls.append(1)
        return int


class Plain:
    """No override: the miss path reads the builtin getset and answers False."""


class Counted:
    """The getter mutates state the caller can observe afterwards."""

    ticks = 0

    @property
    def __class__(self):
        Counted.ticks += 1
        return str


def masked_loop(n):
    hits = 0
    i = 0
    while i < n:
        if isinstance(Masked(), int):
            hits = hits + 1
        i = i + 1
    return hits


def plain_loop(n):
    misses = 0
    i = 0
    while i < n:
        if not isinstance(Plain(), int):
            misses = misses + 1
        i = i + 1
    return misses


def counted_loop(n):
    hits = 0
    i = 0
    while i < n:
        if isinstance(Counted(), str):
            hits = hits + 1
        i = i + 1
    return hits


# ── a property-masked class answers True, and the getter runs every time ──
print("masked hits", masked_loop(N))
print("masked getter calls", len(calls))

# ── the ordinary class still takes the builtin read and answers False ──
print("plain misses", plain_loop(N))

# ── the getter's side effect is observable after the loop ──
print("counted hits", counted_loop(N))
print("counted ticks", Counted.ticks)

# ── the same instance re-tested keeps calling the getter ──
one = Masked()
before = len(calls)
same = 0
for _ in range(N):
    if isinstance(one, int):
        same = same + 1
print("repeat hits", same)
print("repeat getter calls", len(calls) - before)

# ── isinstance against a non-matching type still consults the property ──
before = len(calls)
other = 0
for _ in range(N):
    if isinstance(one, dict):
        other = other + 1
print("dict hits", other)
print("dict getter calls", len(calls) - before)

print("OK")
