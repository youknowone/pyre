# pyre-check: gate=1
result = 0
try:
    type()
except TypeError:
    result = 1

assert result == 1
