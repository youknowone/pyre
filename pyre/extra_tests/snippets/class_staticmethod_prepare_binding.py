# pyre-check: gate=1
class Meta(type):
    @staticmethod
    def __prepare__(name, bases):
        return {'seed': 41}
class C(metaclass=Meta):
    value = seed + 1
result = C.value

assert result == 42
