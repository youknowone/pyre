# pyre-check: max-pypy-ratio=6
# A user metaclass whose __new__/__init__ takes **kw receives class keywords
# through the by-name keyword binder (resolve_kwargs).  A lone-surrogate class
# keyword must survive as a byte-ish name instead of crashing the strict-UTF-8
# accessor.

S1 = '\udc81'
S2 = '\udc84'


def main():
    seen = []

    class Meta(type):
        def __new__(mcs, name, bases, ns, **kw):
            seen.extend(sorted(kw.items(), key=lambda kv: [ord(c) for c in kv[0]]))
            return super().__new__(mcs, name, bases, ns)

    class C(metaclass=Meta, **{S1: 7, S2: 8}):
        pass

    print('meta_kw', [([ord(c) for c in k], v) for k, v in seen])

    # A plain str class keyword alongside the surrogates still binds.
    seen.clear()

    class D(metaclass=Meta, **{S1: 1, 'plain': 2}):
        pass

    print('mixed_kw', sorted((k if k.isascii() else 'S', v) for k, v in seen))


main()
