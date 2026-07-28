# pyre-check: max-pypy-ratio=5
# descroperation.py:234 — the type getattribute slot is wrapped so even the
# hardcoded __abstractmethods__ AttributeError (raised when the slot is unset)
# consults the metaclass __getattr__ before propagating.


class Meta(type):
    def __getattr__(cls, name):
        return 'meta_hook:' + name


class C(metaclass=Meta):
    pass


def main():
    # __abstractmethods__ is unset on C, so the metaclass __getattr__ answers
    print('abstractmethods', C.__abstractmethods__)
    # a normal missing name takes the same fallback
    print('missing', C.whatever)


main()
