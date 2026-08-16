# pyre-check: gate=1
class Direct(type):
    def __setattr__(cls, name, value):
        type.__setattr__(cls, name, value)
class Plain:
    pass
class DirectMeta(Plain, Direct):
    pass
direct = DirectMeta('DirectClass', (object,), {})
direct.answer = 42
class Indirect(type):
    def __setattr__(cls, name, value):
        object.__setattr__(cls, name, value)
class IndirectMeta(Plain, Indirect):
    pass
indirect = IndirectMeta('IndirectClass', (object,), {})
try:
    indirect.answer = 42
except TypeError:
    rejected = True
else:
    rejected = False
result = direct.answer == 42 and rejected

assert result
