# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
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
