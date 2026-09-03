# CPython-suite gap: format tests call a bad __format__ once, never from a compiled loop.
# parity-tests reason: pins the __format__ result check across a type change the body does not branch on.

"""A compiled FORMAT_WITH_SPEC rejects a __format__ answer that is not a str."""


try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


# `DescrOperation.format` raises after the call when `__format__` hands back a
# non-string.  A route that inlines the method body in place of the residual
# helper owes that check, and only a loop long enough to compile shows whether
# it kept it.  This arm is answered while the trace is being recorded, so it
# covers the record-time half.
class AlwaysBad:
    def __format__(self, spec):
        return 42


obj = AlwaysBad()
raised = 0
for _ in range(30000):
    try:
        f"{obj:>4}"
    except TypeError:
        raised += 1
assert raised == 30000, raised


# The half a record-time check cannot cover: the body returns an attribute it
# does not branch on, so nothing in the recorded trace pins the result's type
# and a later iteration can hand back a non-string through the same path.  The
# read is idempotent, so re-running the opcode on a deopt answers the same.
class Attr:
    def __format__(self, spec):
        return self.out


obj = Attr()
obj.out = "s"
raised = 0
for i in range(30000):
    if i == 15000:
        obj.out = 42
    try:
        f"{obj:>4}"
    except TypeError:
        raised += 1
assert raised == 15000, raised


# The same shape with the string arm second: a trace recorded on the failing
# answer must not keep raising once the attribute holds a string again.
obj = Attr()
obj.out = 42
raised = 0
for i in range(30000):
    if i == 15000:
        obj.out = "t"
    try:
        seen = f"{obj:>4}"
    except TypeError:
        raised += 1
    else:
        assert seen == "t", seen
assert raised == 15000, raised

print("OK")
