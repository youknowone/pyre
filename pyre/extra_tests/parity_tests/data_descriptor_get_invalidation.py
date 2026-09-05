# CPython-suite gap: no suite test mutates the class of an installed user data
# descriptor after LOAD_ATTR has compiled its Python __get__ body.
# parity-tests reason: this is a JIT descriptor-shape invalidation regression.

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 5000


class One:
    def __get__(self, obj, owner):
        return 1

    def __set__(self, obj, value):
        pass


class Two:
    def __get__(self, obj, owner):
        return 2

    def __set__(self, obj, value):
        pass


descriptor = One()


class Owner:
    value = descriptor


owner = Owner()
total = 0
for i in range(N):
    if i == N // 2:
        # BaseUserClassMapdict.setclass replaces the descriptor's map.  A trace
        # may keep the installed descriptor identity, but must stop using the
        # __get__ selected through its old map/class.
        descriptor.__class__ = Two
    total += owner.value

assert total == N + N // 2

# W_TypeObject.mutated invalidates the promoted lookup independently of the
# descriptor-map guard when the class stays fixed and its method is rebound.
Two.__get__ = lambda self, obj, owner: 3
total = 0
for _ in range(N):
    total += owner.value
assert total == 3 * N

print("OK")
