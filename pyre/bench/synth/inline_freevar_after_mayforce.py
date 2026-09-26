# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=<module>,entry-bridge:forward
# The module loop inlines `forward`, whose Ratio division is a forcing call.
# The following LOAD_DEREF must recover `adjust` from the inlined callee's own
# frame shadow after that call invalidates heap-cache facts.  Losing the callee
# frame makes the walk abort or resume with the wrong closure cell.
#
# The second entry was `root:_div` until `Fraction._div` stopped taking a
# function-entry trace of its own and started being inlined into its caller
# instead.  That is a different trace for the same code, not a lost one: the
# division still forces -- the run emits more `ForceToken` than it did, not
# fewer -- and `forward`, the callee this guard is actually about, compiles on
# both sides of the change.
#
# `Ratio` keeps the shape of `fractions.Fraction` this needs -- `__truediv__`
# built by `_operator_fallbacks` around a monomorphic `_div`, a normalizing
# `__new__`, and the int comparison -- without importing `fractions`, which
# pulls in `math`.


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


class Ratio:
    __slots__ = ("_numerator", "_denominator")

    def __new__(cls, numerator, denominator):
        g = _gcd(numerator, denominator)
        if denominator < 0:
            g = -g
        self = object.__new__(cls)
        self._numerator = numerator // g
        self._denominator = denominator // g
        return self

    def _operator_fallbacks(monomorphic_operator):
        def forward(a, b):
            if isinstance(b, Ratio):
                return monomorphic_operator(a, b)
            return NotImplemented

        return forward

    def _div(a, b):
        return Ratio(a._numerator * b._denominator, a._denominator * b._numerator)

    __truediv__ = _operator_fallbacks(_div)

    def __gt__(a, b):
        return a._numerator > b * a._denominator

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


# Keep object construction outside the measured loop.  The regression needs
# one forcing Ratio division followed by LOAD_DEREF of `adjust`; rebuilding
# three invariant Ratios per iteration only measures constructor overhead
# and does not add another resume shape.
DIVISOR = Ratio(2, 89)
INPUTS = tuple(Ratio(i + 1, 97) for i in range(97))


# Low thresholds make both required compilation arms deterministic; additional
# iterations only repeat Ratio allocation and no longer strengthen the test.
N = 6000


def make_forwarder():
    def adjust(value):
        return value

    def forward(value):
        scaled = value / DIVISOR
        return adjust(scaled)

    return forward


forward = make_forwarder()
count = 0
for i in range(N):
    value = forward(INPUTS[i % 97])
    if value > 1:
        count += 1

cycles, tail = divmod(N, 97)
expected = cycles * 95 + max(0, tail - 2)
if count != expected:
    raise AssertionError(f"closure result mismatch: {count} != {expected}")
print("PASS inline freevar after forcing call")
