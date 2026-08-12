# CPython-suite gap: the suite checks float-subclass arithmetic and coercion,
# not whether a subclass instance survives storage in a container.
# parity-tests reason: pyre's raw-f64 storage reboxes on read as an exact float,
# which only `type()` / `is` sees.

# `FloatListStrategy.is_correct_type`, `AbstractAttribute._pick_unbox_type` and
# `makespecialisedtuple2` are all strict `type(w) is W_FloatObject`, so a float
# SUBCLASS instance must keep its box.  pyre gives a subclass the same builtin
# `ob_type` and separates it only by `w_class`, so every JIT fast path that
# unboxes into raw-f64 storage needs a `w_class` guard, not just a class guard:
# a class guard alone lets a subclass reuse a trace recorded for exact floats.

f = 1.5


class F(float):
    pass


s = F(2.5)

assert type(s) is F
assert s == 2.5
assert type(f) is float

# --- arity-2 tuple: `makespecialisedtuple2` / Cls_ff -----------------------
t = (s, s)
assert t[0] is s and t[1] is s

# One subclass slot is enough — both slots are stored raw.
mixed = (f, s)
assert mixed[0] is f and mixed[1] is s

# --- instance attribute: mapdict `UnboxedPlainAttribute` -------------------
class C:
    pass


c = C()
c.x = s
assert c.x is s and c.__dict__["x"] is s

# An existing unboxed float slot must convert back to boxed storage.
c2 = C()
c2.y = f
c2.y = s
assert c2.y is s

# --- list: FloatListStrategy / IntOrFloatListStrategy ----------------------
lst = [s]
assert lst[0] is s

lst2 = [1.0, 2.0]
lst2.append(s)
assert lst2[2] is s and type(lst2[0]) is float

lst3 = [1, 2.0]
lst3.append(s)
assert lst3[2] is s

lst4 = [1.0, 2.0]
lst4[0] = s
assert lst4[0] is s

lst5 = []
lst5.append(s)
assert lst5[0] is s


# Hand each finite-float trace a subclass instance on its final iteration.
# A guard failure exits the whole loop iteration, so each direct store needs
# its own loop.
def warm_attr(rounds):
    obj = C()
    obj.x = 0.5
    for i in range(rounds):
        v = s if i == rounds - 1 else i * 0.5
        obj.x = v
    return obj.x is s


def warm_setitem(rounds):
    lst = [0.5]
    for i in range(rounds):
        v = s if i == rounds - 1 else i * 0.5
        lst[0] = v
    return lst[0] is s


def warm_newlist(rounds):
    for i in range(rounds):
        v = s if i == rounds - 1 else i * 0.5
        got = [v]
    return got[0] is s


def warm_append_empty(rounds):
    for i in range(rounds):
        v = s if i == rounds - 1 else i * 0.5
        got = []
        got.append(v)
    return got[0] is s


def warm_append_float(rounds):
    for i in range(rounds):
        v = s if i == rounds - 1 else i * 0.5
        got = [0.5]
        got.append(v)
    return got[1] is s


assert warm_attr(3000)
assert warm_setitem(3000)
assert warm_newlist(3000)
assert warm_append_empty(3000)
assert warm_append_float(3000)

print("OK")
