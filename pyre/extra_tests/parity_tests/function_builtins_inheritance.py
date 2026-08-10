import types


def sample(value):
    return len(value)


empty_globals = {}
inherited = types.FunctionType(sample.__code__, empty_globals)
builtins_dict = __builtins__.__dict__ if hasattr(__builtins__, "__dict__") else __builtins__
assert inherited.__globals__ is empty_globals
assert inherited.__builtins__ is builtins_dict
assert inherited("abc") == 3
assert empty_globals == {}

safe_builtins = {"None": None}
namespace = {"type": type, "__builtins__": safe_builtins}
exec(
    "def inner(): pass\n"
    "cloned = type(inner)(inner.__code__, {})\n",
    namespace,
)
assert namespace["inner"].__builtins__ is safe_builtins
assert namespace["cloned"].__builtins__ is safe_builtins
assert "__builtins__" not in namespace["cloned"].__globals__


class GlobalsSubclass(dict):
    def __getitem__(self, key):
        raise AssertionError("function construction must use the dict backing")


custom_builtins = {"marker": object()}
subclass_globals = GlobalsSubclass(__builtins__=custom_builtins)
subclass_function = types.FunctionType(sample.__code__, subclass_globals)
assert subclass_function.__builtins__ is custom_builtins

print("OK")
