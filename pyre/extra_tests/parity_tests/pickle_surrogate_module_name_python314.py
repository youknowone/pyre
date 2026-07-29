"""CPython 3.14 module names containing a lone surrogate."""

import pickle
import sys
import types


class Reduced:
    def __reduce__(self):
        return "value"


module_name = "nonencodable\udbff"
value = Reduced()
value.__module__ = module_name
module = types.SimpleNamespace(value=value)
sys.modules[module_name] = module

try:
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        if protocol < 4:
            try:
                pickle.dumps(value, protocol=protocol)
            except pickle.PicklingError as exc:
                assert str(exc) == (
                    f"can't pickle module identifier {module_name!r} "
                    f"using pickle protocol {protocol}"
                )
                assert isinstance(exc.__context__, UnicodeEncodeError)
            else:
                raise AssertionError(
                    f"surrogate module name was encoded by protocol {protocol}"
                )
        else:
            restored = pickle.loads(pickle.dumps(value, protocol=protocol))
            assert restored is value
finally:
    del sys.modules[module_name]

print("OK")
