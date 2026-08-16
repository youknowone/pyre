# pyre-check: gate=1
n = 10000
a = b'a' * n
b = b'b' * n
haystack = a + a + b + a + a
needle = a + b + b + a
result = True
for constructor in (bytes, bytearray):
    value = constructor(haystack)
    sub = constructor(needle)
    result = (
        result
        and value.find(sub) == -1
        and value.rfind(sub) == -1
        and value.count(sub) == 0
        and (value + sub).find(sub) == len(value)
        and (value + sub).count(sub) == 1
    )

assert result
