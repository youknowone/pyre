# CPython-suite gap: the suite checks NaN value semantics, not identity through containers.
# parity-tests reason: pyre's raw-f64 storage reboxes on read, which only `is` / `id()` sees.

# Python 3.14 gives NaNs pointer identity. Raw-f64 list, tuple, and mapdict
# storage must therefore reject them instead of reboxing on read. These checks
# cover the interpreter and JIT paths; finite floats must remain eligible.

n = float("nan")
m = float("nan")
f = 1.5

assert n is not m
assert id(n) != id(m)
assert n is n and n != n

# --- arity-2 tuple: `makespecialisedtuple2` / Cls_ff -----------------------
t = (n, n)
assert t[0] is n and t[1] is n

# One NaN makes the whole pair ineligible for `Cls_ff`.
mixed = (f, n)
assert mixed[0] is f and mixed[1] is n

# --- instance attribute: mapdict `UnboxedPlainAttribute` -------------------
class C:
    pass


c = C()
c.x = n
assert c.x is n and c.__dict__["x"] is n

# An existing float slot must convert back to boxed storage.
c2 = C()
c2.y = f
c2.y = n
assert c2.y is n

# --- list: FloatListStrategy / IntOrFloatListStrategy ----------------------
lst = [n]
assert lst[0] is n

lst2 = [1.0, 2.0]
lst2.append(n)
assert lst2[2] is n

lst3 = [1, 2.0]
lst3.append(n)
assert lst3[2] is n

# `list.index`/`count` compare by value; a NaN is found only via the identity
# shortcut `==` gets before the value compare.
assert lst.index(n) == 0
assert lst.count(n) == 1
assert n in lst


# Hand each finite-float trace a NaN on its final iteration. A guard failure
# exits the whole loop iteration, so each direct store needs its own loop.
def warm_attr(rounds):
    obj = C()
    obj.x = 0.5
    for i in range(rounds):
        v = n if i == rounds - 1 else i * 0.5
        obj.x = v
    return obj.x is n


def warm_setitem(rounds):
    lst = [0.5]
    for i in range(rounds):
        v = n if i == rounds - 1 else i * 0.5
        lst[0] = v
    return lst[0] is n


def warm_newlist(rounds):
    for i in range(rounds):
        v = n if i == rounds - 1 else i * 0.5
        got = [v]
    return got[0] is n


def warm_append_empty(rounds):
    for i in range(rounds):
        v = n if i == rounds - 1 else i * 0.5
        got = []
        got.append(v)
    return got[0] is n


def warm_append_float(rounds):
    for i in range(rounds):
        v = n if i == rounds - 1 else i * 0.5
        got = [0.5]
        got.append(v)
    return got[1] is n


assert warm_attr(3000)
assert warm_setitem(3000)
assert warm_newlist(3000)
assert warm_append_empty(3000)
assert warm_append_float(3000)

print("OK")
