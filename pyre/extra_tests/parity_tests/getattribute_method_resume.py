# CPython-suite gap: no suite test resumes a method-form LOAD_ATTR while its
# inlined custom __getattribute__ frame is live.
# parity-tests reason: this is a JIT caller-stack reconstruction regression.

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 5000
use_other = [False]


class C:
    def f(*args):
        return len(args)

    def g(*args):
        return 10 + len(args)

    def __getattribute__(self, name):
        if use_other[0]:
            return type(self).__dict__["g"]
        return type(self).__dict__[name]


c = C()
total = 0
for i in range(N):
    if i == N // 2:
        use_other[0] = True
    # execute_load_attr's method form replaces `c` with `(attr,
    # self_or_null)`.  A guard in the inlined
    # DescrOperation._handle_getattribute call must rebuild the caller at the
    # first residual's return point, before compute_load_method_bound produces
    # the second result.
    total += c.f(i)

assert total == 30000
print("OK")
