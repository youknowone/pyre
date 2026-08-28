"""A resumed inlined closure must retain both of its freevar cells."""


def run():
    first = -1074
    second = 53

    def pair(value):
        return first, second

    result = None
    for value in range(10_000):
        result = pair(value)
    return result


# A two-object closure uses PyPy's W_SpecialisedTupleObject_oo storage.  The
# guard failure that ends the hot loop must materialize the inlined frame from
# its two inline tuple fields, rather than treating value0 as `wrappeditems`.
assert run() == (-1074, 53)
