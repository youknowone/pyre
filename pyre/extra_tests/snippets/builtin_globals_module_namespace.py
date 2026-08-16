# pyre-check: gate=1
x = 41
result = globals()['x'] + 1

assert result == 42
