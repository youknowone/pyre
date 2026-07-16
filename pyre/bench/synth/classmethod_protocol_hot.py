"""Hot classmethod binding and wrapper-dictionary parity."""


class Base:
    @classmethod
    def calculate(cls, value):
        return value + len(cls.__name__)


class Derived(Base):
    pass


wrapper = Base.__dict__["calculate"]
wrapper.offset = 3

total = 0
for i in range(1001):
    total += Derived.calculate(i) + wrapper.offset

print(total)
print(callable(wrapper))
print(wrapper.__func__.__name__, wrapper.__dict__["offset"])
