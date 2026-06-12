N = 200000


class MyExit(SystemExit):
    pass


class MyOS(OSError):
    pass


class MyErr(Exception):
    def __str__(self):
        return "custom-str:" + repr(self.args)

    def __repr__(self):
        return "custom-repr"


def main():
    # The typed exception attributes (`code`, `errno`, `args`, ...) are
    # data descriptors whose `__set__` must win over the instance __dict__
    # a user subclass carries — and a subclass `__str__`/`__repr__` override
    # must shadow the builtin exception formatting.  Run both in a hot loop
    # so the JIT path exercises the same dispatch as the interpreter.
    code_sum = 0
    errno_sum = 0
    arglen = 0
    str_hits = 0
    repr_hits = 0
    i = 0
    while i < N:
        m = MyExit(i)
        m.code = i + 1
        code_sum = code_sum + m.code

        o = MyOS(2, "msg")
        o.errno = i % 7
        errno_sum = errno_sum + o.errno

        e = MyErr("a")
        e.args = (1, 2, 3)
        arglen = arglen + len(e.args)
        if str(e) == "custom-str:(1, 2, 3)":
            str_hits = str_hits + 1
        if repr(e) == "custom-repr":
            repr_hits = repr_hits + 1
        i = i + 1
    print(code_sum, errno_sum, arglen, str_hits, repr_hits)


main()
