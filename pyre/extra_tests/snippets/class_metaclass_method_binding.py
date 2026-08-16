# pyre-check: gate=1
class Meta(type):
    def pick(cls):
        return cls
class C(metaclass=Meta):
    pass
bound = C.pick
result = bound()

assert result is C
