import types


code = compile("superglobal", "test", "eval")
namespace = {"__builtins__": types.MappingProxyType({"superglobal": 1})}
assert eval(code, namespace) == 1

namespace = {"__builtins__": types.MappingProxyType({})}
try:
    eval(code, namespace)
except NameError:
    pass
else:
    raise AssertionError("missing custom builtin did not raise NameError")

namespace = {"__builtins__": types.MappingProxyType({}), "x": iter([1, 2])}
try:
    eval(compile("x.__reduce__()", "test", "eval"), namespace)
except AttributeError as error:
    assert "iter" in str(error)
else:
    raise AssertionError("iterator reduce ignored the selected builtin mapping")

try:
    exec(compile("class A: pass", "test", "exec"), {"__builtins__": {}})
except NameError as error:
    assert "__build_class__" in str(error)
else:
    raise AssertionError("LOAD_BUILD_CLASS ignored the selected builtin mapping")


class LocalsMapping:
    def __getitem__(self, key):
        raise KeyError(key)

    def keys(self):
        return list("xyz")


assert eval("dir()", globals(), LocalsMapping()) == list("xyz")
print("OK")
