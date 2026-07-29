"""CPython 3.14 global names containing a lone surrogate."""

import pickle


class Reduced:
    def __reduce__(self):
        return self.name


name = "nonencodable\udbff"
value = Reduced()
value.name = name
value.__module__ = __name__
globals()[name] = value

try:
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        if protocol < 4:
            try:
                pickle.dumps(value, protocol=protocol)
            except pickle.PicklingError as exc:
                assert str(exc) == (
                    f"can't pickle global identifier {name!r} "
                    f"using pickle protocol {protocol}"
                )
                assert isinstance(exc.__context__, UnicodeEncodeError)
            else:
                raise AssertionError(
                    f"surrogate global name was encoded by protocol {protocol}"
                )
        else:
            restored = pickle.loads(pickle.dumps(value, protocol=protocol))
            assert restored is value
finally:
    del globals()[name]

print("OK")
