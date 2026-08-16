# pyre-check: gate=1
MT = type(__builtins__)
try:
    class Module(MT, str):
        pass
except TypeError:
    result = True
else:
    result = False

assert result
