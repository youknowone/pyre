import io
import pickle
import sys
import types


class NewObj:
    def __new__(cls):
        return super().__new__(cls)


payload = pickle.dumps(NewObj(), protocol=4)
old_newobj = globals()["NewObj"]
globals()["NewObj"] = 42
try:
    try:
        pickle._Unpickler(io.BytesIO(payload)).load()
    except (TypeError, pickle.UnpicklingError):
        pass
    else:
        raise AssertionError("NEWOBJ accepted a non-type")
finally:
    globals()["NewObj"] = old_newobj


name = "nonencodable\udbff"
module = types.SimpleNamespace(value=42)
missing = object()
old_module = sys.modules.get(name, missing)
sys.modules[name] = module
try:
    assert __import__(name) is module
finally:
    if old_module is missing:
        del sys.modules[name]
    else:
        sys.modules[name] = old_module


for factory, depth in (
    (lambda value: frozenset((1, value)), 100),
):
    value = None
    for _ in range(depth):
        value = factory(value)
    buffer = io.BytesIO()
    pickler = pickle._Pickler(buffer, protocol=0)
    pickler.fast = True
    pickler.dump(value)
    assert pickle._Unpickler(io.BytesIO(buffer.getvalue())).load() == value

print("OK")
