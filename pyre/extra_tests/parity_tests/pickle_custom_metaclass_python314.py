"""CPython 3.14 custom-metaclass dispatch-table pickle parity."""

import copyreg
import pickle


class PicklingMeta(type):
    def __reduce__(self):
        return rebuild, (self.__name__, self.__bases__)


def rebuild(name, bases):
    return PicklingMeta(name, bases, {})


copyreg.pickle(PicklingMeta, PicklingMeta.__reduce__)
dynamic = PicklingMeta("Dynamic", (object,), {})

for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    restored = pickle.loads(pickle.dumps(dynamic, protocol))
    assert isinstance(restored, PicklingMeta)
    assert restored.__name__ == dynamic.__name__
    assert restored.__bases__ == dynamic.__bases__

print("OK")
