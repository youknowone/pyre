def m(label, value):
    print(label, "->", repr(value))


for _ in range(2000):
    group = ExceptionGroup("warmup", [ValueError(1), TypeError(2)])
    group.split(ValueError)

group = ExceptionGroup("g", [ValueError(1), TypeError(2)])
m("type", type(group).__name__)
m("mro", [cls.__name__ for cls in type(group).__mro__])
m("message", group.message)
m("exceptions", [type(exc).__name__ for exc in group.exceptions])
m("split", group.split(ValueError))
m("subgroup", group.subgroup(ValueError))
m("derive", group.derive([KeyError()]))
m("base promotion", type(BaseExceptionGroup("g", [ValueError()])).__name__)
m("base hierarchy", issubclass(ExceptionGroup, BaseExceptionGroup))
m("string", str(group))
m("representation", repr(group))
