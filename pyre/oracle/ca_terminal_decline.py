def inner(limit):
    i = 0
    total = 0
    while i < limit:
        total = total + i * 3 + 1
        i = i + 1
    return total


def outer(count, limit):
    i = 0
    total = 0
    while i < count:
        total = total + inner(limit)
        i = i + 1
    return total


print(outer(3000, 31))
