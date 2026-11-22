# CPython-suite gap: no suite test assigns a type's slot wrapper
# (`tuple.__hash__`, `list.__len__`) onto another class and then calls
# hash()/len() on an instance of that class, or hashes a tuple subclass
# through set membership.
# parity-tests reason: get_and_call_function's Function/FunctionWithFixedCode
# fast path must include SLOT_WRAPPER_TYPE so a typedef-owned dunder is not
# bound; a foreign receiver still raises TypeError naming the owning type,
# and a tuple subclass still hashes as a tuple.

"""Slot wrappers refuse a foreign receiver and accept a subclass."""

class C:
    __hash__ = tuple.__hash__
class D:
    __len__ = list.__len__
class E(tuple):
    pass
for f, o in ((hash, C()), (len, D())):
    try:
        print(f(o))
    except TypeError as e:
        print("TypeError", e)
print(hash(E((1, 2))) == hash((1, 2)))
print(len({(1, 2), E((1, 2))}))
print("OK")
