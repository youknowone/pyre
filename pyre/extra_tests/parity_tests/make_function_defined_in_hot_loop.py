# CPython-suite gap: no suite test defines a function inside a loop hot enough
# to trace and then reads back the slots SET_FUNCTION_ATTRIBUTE stamped on it.
# parity-tests reason: the assertions target pyre's MAKE_FUNCTION /
# SET_FUNCTION_ATTRIBUTE trace lowering, which emits the definition sequence as
# an allocation plus typed field stores rather than opaque calls.

# A `def` in a hot loop body is rebuilt on every iteration, and the opcodes that
# follow it stamp `__defaults__`, `__kwdefaults__` and `__closure__` onto that
# fresh function.  The trace emits those stores directly on the allocation, so
# what has to hold is that the slots still read back as the objects the
# definition expressions produced -- by identity, not just by value -- whether
# the function is consumed inside the iteration or escapes it.
#
# N puts every loop below past the trace threshold; at 300 the specialization
# census reports `consulted=0`, so the assertions would cover only the
# interpreter.

N = 3000


def check(got, want, label):
    assert got == want, "%s: %r != %r" % (label, got, want)


# ── the stamped tuple is the one the expression produced, read back by
#    identity through `__defaults__` ──
def defaults_identity_loop():
    seen = 0
    for i in range(N):
        marker = (i, i + 1)

        def take(a=marker):
            return a

        got = take.__defaults__
        assert got is not None, "no defaults"
        assert len(got) == 1, "arity"
        assert got[0] is marker, "defaults element is not the built object"
        assert take() is marker, "call did not bind the same object"
        seen += 1
    return seen


check(defaults_identity_loop(), N, "defaults identity")


# ── keyword-only defaults land in a dict, and it is the same dict the
#    definition built: mutating it through `__kwdefaults__` is visible ──
def kwdefaults_loop():
    total = 0
    for i in range(N):
        def kw(*, k=i):
            return k

        d = kw.__kwdefaults__
        check(d, {"k": i}, "kwdefaults value")
        d["k"] = i + 100
        total += kw()
    return total


check(kwdefaults_loop(), sum(i + 100 for i in range(N)), "kwdefaults mutation")


# ── the function escapes the iteration that built it, so the allocation is
#    materialized and every slot has to survive on the real object ──
def escaping_loop():
    made = []
    for i in range(N):
        tag = "f%d" % i

        def both(a, b=i, *, k=tag):
            return (a, b, k)

        made.append(both)
    return made


funcs = escaping_loop()
check(len(funcs), N, "escaped count")
for index, fn in enumerate(funcs):
    check(fn.__defaults__, (index,), "escaped defaults %d" % index)
    check(fn.__kwdefaults__, {"k": "f%d" % index}, "escaped kwdefaults %d" % index)
    check(fn(1), (1, index, "f%d" % index), "escaped call %d" % index)
    check(fn.__name__, "both", "escaped name %d" % index)
    check(fn.__qualname__, "escaping_loop.<locals>.both", "escaped qualname %d" % index)


# ── an annotated def: 3.14 stamps `__annotate__` before the defaults, so this
#    is the shape whose first stamp decides whether the rest folds at all ──
def annotated_loop():
    total = 0
    for i in range(N):
        def add(value: int = i) -> int:
            return value + 1

        check(add.__annotations__, {"value": int, "return": int}, "annotations")
        total += add()
    return total


check(annotated_loop(), sum(i + 1 for i in range(N)), "annotated")


# ── annotations only, no defaults ──
def annotations_without_defaults_loop():
    total = 0
    for i in range(N):
        def twice(value: int) -> int:
            return value * 2

        assert twice.__defaults__ is None, "annotated def gained defaults"
        check(twice.__annotations__, {"value": int, "return": int}, "ann-only")
        total += twice(i)
    return total


check(annotations_without_defaults_loop(), sum(i * 2 for i in range(N)), "ann only")


# ── the annotations of a function that escapes its iteration ──
def escaping_annotated_loop():
    made = []
    for i in range(N):
        def tagged(a, b: str = "s") -> bool:
            return (a, b)

        made.append(tagged)
    return made


annotated_funcs = escaping_annotated_loop()
check(len(annotated_funcs), N, "escaped annotated count")
for fn in annotated_funcs:
    check(fn.__annotations__, {"b": str, "return": bool}, "escaped annotations")
    check(fn.__defaults__, ("s",), "escaped annotated defaults")
    check(fn(1), (1, "s"), "escaped annotated call")


# ── rebinding `__annotations__` on a function the loop just built ──
def rebound_annotations_loop():
    seen = 0
    for i in range(N):
        def add(value: int = 0) -> int:
            return value

        add.__annotations__ = {"value": str}
        check(add.__annotations__, {"value": str}, "rebound annotations")
        seen += 1
    return seen


check(rebound_annotations_loop(), N, "rebound annotations")

print("OK")
