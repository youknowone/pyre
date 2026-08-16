# pyre-check: gate=1
result = 0
try:
    vars(1)
except TypeError:
    result = 1

assert result == 1
