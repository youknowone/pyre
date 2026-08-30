# A float subclass that overrides the arithmetic and comparison dunders, driven
# hot enough to compile. The walker's float specialization lowers BINARY_OP to
# `FloatAdd` / `FloatSub` / `FloatMul` / `FloatTrueDiv` and COMPARE_OP to
# `FloatLt` / `FloatEq` / ... , all of which bypass special-method dispatch.
#
# A numeric subclass keeps the builtin `ob_type` layout while its Python-visible
# class lives in `w_class`, so the `guard_class` those paths emit reads `ob_type`
# and cannot tell the subclass apart at runtime either. Only an exactness test on
# the concrete operands keeps a subclass out of the raw path, which is what
# `walker_float_specialization_operands` checks before returning its operands.
# Without it every line below silently loses the override and prints the raw
# IEEE result.
#
# The int subclass at the bottom is the control for the record-time gate: when
# the subclass is present from the first iteration, the gate sees it on the
# recorded operand and declines, and every fold below stayed correct on that
# shape alone.
#
# That shape is not sufficient. The `warm_then_swap_*` cases below compile the
# trace from EXACT builtins first and introduce the subclass afterwards, so the
# gate never sees it and only the emitted guard can reject it. `compare_op_int`,
# `compare_op_float`, `store_subscr`, `newlist` and the `store_attr` in-place
# arm each emitted the `ob_type` unbox guard without the matching `w_class` pin
# and answered these with the raw payload -- `a < 1` returning True where the
# override returns a string, and a stored subclass reading back as a plain int.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 1000


class MyFloat(float):
    def __add__(self, other):
        return float.__add__(self, other) + 1000.0

    def __radd__(self, other):
        return float.__radd__(self, other) + 2000.0

    def __sub__(self, other):
        return float.__sub__(self, other) - 1000.0

    def __mul__(self, other):
        return float.__mul__(self, other) * 2.0

    def __truediv__(self, other):
        return float.__truediv__(self, other) + 7.0

    def __lt__(self, other):
        return not float.__lt__(self, other)

    def __eq__(self, other):
        return not float.__eq__(self, other)

    __hash__ = float.__hash__


def add_hot(n):
    total = 0.0
    for i in range(n):
        total = MyFloat(1.5) + total
    return total


def radd_hot(n):
    total = 0.0
    for i in range(n):
        total = total + MyFloat(1.5)
    return total


def sub_hot(n):
    # Accumulating rather than alternating: `MyFloat(1.5) - total` oscillates
    # between two values and lands on the same result with or without the
    # override, so it would not discriminate.
    total = 0.0
    for i in range(n):
        total = MyFloat(total) - 1.0
    return total


def mul_hot(n):
    total = 0.0
    for i in range(n):
        total = MyFloat(2.0) * 1.5
    return total


def truediv_hot(n):
    total = 0.0
    for i in range(n):
        total = MyFloat(9.0) / 2.0
    return total


def lt_hot(n):
    hits = 0
    for i in range(n):
        if MyFloat(1.0) < 2.0:
            hits += 1
    return hits


def eq_hot(n):
    hits = 0
    for i in range(n):
        if MyFloat(1.0) == 1.0:
            hits += 1
    return hits


# Mixed operands: an exact float on one side, the subclass on the other, so the
# int/float coercion arm is exercised as well as the plain float/float arm.
def mixed_int_operand_hot(n):
    total = 0.0
    for i in range(n):
        total = MyFloat(3.0) + 2
    return total


class MyInt(int):
    def __add__(self, other):
        return int.__add__(self, other) + 1000


def int_control_hot(n):
    total = 0
    for i in range(n):
        total = MyInt(3) + total
    return total


print(add_hot(N))
print(radd_hot(N))
print(sub_hot(N))
print(mul_hot(N))
print(truediv_hot(N))
print(lt_hot(N))
print(eq_hot(N))
print(mixed_int_operand_hot(N))
print(int_control_hot(N))


# --- warm on the exact builtin, then swap in the subclass -------------------
# The list has no branch on the element, so the compiled trace can only reject
# the tail element through a type guard. Each function prints what the override
# says; a fold missing its `w_class` pin prints the raw builtin answer instead.
class LiarInt(int):
    def __lt__(self, other):
        return "LT"


class LiarFloat(float):
    def __lt__(self, other):
        return "FLT"


class Slotted:
    __slots__ = ("x",)


class LiarBool(int):
    def __bool__(self):
        return True


def warm_then_swap_compare_int(n):
    out = None
    for a in [0] * n + [LiarInt(0)]:
        out = a < 1
    return out


def warm_then_swap_compare_float(n):
    out = None
    for a in [0.0] * n + [LiarFloat(0.0)]:
        out = a < 1.0
    return out


def warm_then_swap_store_subscr(n):
    lst = [0]
    for a in [0] * n + [LiarInt(7)]:
        lst[0] = a
    return type(lst[0]).__name__


def warm_then_swap_newlist(n):
    out = None
    for a in [0] * n + [LiarInt(7)]:
        out = [a]
    return type(out[0]).__name__


def warm_then_swap_store_attr(n):
    holder = Slotted()
    for a in [0] * n + [LiarInt(7)]:
        holder.x = a
    return type(holder.x).__name__


# The same three storage paths on the float side. The value's exact class must
# survive the store: a `LiarFloat` written through a warmed-up float store must
# read back as `LiarFloat`, not as `float`.
def warm_then_swap_store_subscr_float(n):
    lst = [0.0]
    for a in [0.0] * n + [LiarFloat(7.0)]:
        lst[0] = a
    return type(lst[0]).__name__


def warm_then_swap_newlist_float(n):
    out = None
    for a in [0.0] * n + [LiarFloat(7.0)]:
        out = [a]
    return type(out[0]).__name__


def warm_then_swap_store_attr_float(n):
    holder = Slotted()
    for a in [0.0] * n + [LiarFloat(7.0)]:
        holder.x = a
    return type(holder.x).__name__


# Cold start on the storage side: the subclass is the value from the first
# iteration, so the unboxed-strategy gates must decline on the recorded value
# rather than rely on the `w_class` pin to reject a later arrival. An unboxed
# int/float slot stores the raw payload, so a fold here reads back as `int` /
# `float` instead of the subclass.
def store_subscr_cold_subclass(n):
    lst = [0]
    for _ in range(n):
        lst[0] = LiarInt(7)
    return type(lst[0]).__name__


def store_attr_cold_subclass(n):
    holder = Slotted()
    for _ in range(n):
        holder.x = LiarInt(7)
    return type(holder.x).__name__


def store_attr_cold_subclass_float(n):
    holder = Slotted()
    for _ in range(n):
        holder.x = LiarFloat(7.0)
    return type(holder.x).__name__


# `truth_int` reaches the same hole from the branch side rather than the value
# side: `POP_JUMP_IF_*` and the short-circuit operators read the truth of a
# payload the `GUARD_CLASS INT` admits, so a `__bool__` override on a zero-payload
# subclass is skipped and the branch is taken the wrong way. `bool(a)` does not
# reach the fold and stays correct either way, so it is the control.
def warm_then_swap_truth_if(n):
    hits = 0
    for a in [0] * n + [LiarBool(0)]:
        if a:
            hits += 1
    return hits


def warm_then_swap_truth_and(n):
    out = None
    for a in [0] * n + [LiarBool(0)]:
        out = a and "yes"
    return out


# Cold start: the trace records on the subclass from the very first iteration
# rather than meeting it after warming up on exact ints. The fold has to decline
# on the *recorded* operand — pinning `w_class` only rejects a subclass that
# arrives later, and the walk is the authoritative executor, so a payload-folded
# truth here is the answer the program returns. `LiarBool(0)` is falsy by payload
# and true by `__bool__`, so the two answers differ.
def truth_cold_subclass(n):
    hits = 0
    for a in [LiarBool(0)] * n:
        if a:
            hits += 1
    return hits


def truth_bool_call_control(n):
    out = None
    for a in [0] * n + [LiarBool(0)]:
        out = bool(a)
    return out


print(warm_then_swap_compare_int(N))
print(warm_then_swap_compare_float(N))
print(warm_then_swap_store_subscr(N))
print(warm_then_swap_newlist(N))
print(warm_then_swap_store_attr(N))
print(warm_then_swap_store_subscr_float(N))
print(warm_then_swap_newlist_float(N))
print(warm_then_swap_store_attr_float(N))
print(warm_then_swap_truth_if(N))
print(warm_then_swap_truth_and(N))
print(store_subscr_cold_subclass(N))
print(store_attr_cold_subclass(N))
print(store_attr_cold_subclass_float(N))
print(truth_cold_subclass(N))
print(truth_bool_call_control(N))
