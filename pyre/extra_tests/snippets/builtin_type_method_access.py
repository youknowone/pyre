# pyre-check: gate=1
xs = []
m = xs.append
m(42)
result = len(xs)

assert result == 1
