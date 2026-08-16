# pyre-check: gate=1
value = bytearray()
allocations = []
for item in range(20):
    value.append(item)
    allocations.append(value.__alloc__())

del value[:5]
after_prefix_delete = value.__alloc__()
for _ in range(7):
    value.append(0)

class Sub(bytearray):
    pass
sub = Sub(b'abc')

value.clear()
result = (
    allocations == [2, 5, 5, 5, 8, 8, 8, 12, 12, 12, 12,
                    19, 19, 19, 19, 19, 19, 19, 27, 27]
    and after_prefix_delete == 27
    and value.__alloc__() == 1
    and sub.__alloc__() == 4
    and sub.__sizeof__() == Sub.__basicsize__ + sub.__alloc__()
)

assert result
