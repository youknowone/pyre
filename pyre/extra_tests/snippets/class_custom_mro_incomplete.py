# pyre-check: gate=1
observed_none = False
extension_rejected = False
class Meta(type):
    def mro(cls):
        global observed_none, extension_rejected
        observed_none = cls.__mro__ is None
        if observed_none and cls.__name__ == 'C':
            try:
                class Derived(cls):
                    pass
            except TypeError:
                extension_rejected = True
        return type.mro(cls)
class C(metaclass=Meta):
    pass

assert observed_none
assert extension_rejected
