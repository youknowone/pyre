def variable_inner(n):
    total = 0
    for i in range(n):
        outer = i % 5
        for j in range(outer):
            pass
        total += outer
    return total


def constant_inner(n):
    total = 0
    for i in range(n):
        outer = i % 5
        for j in range(3):
            pass
        total += outer
    return total


def single_loop(n):
    total = 0
    for i in range(n):
        local = i % 5
        total += local
    return total


def three_levels(n):
    total = 0
    for i in range(n):
        outer = i % 7
        for j in range(outer):
            middle = j % 3
            for k in range(middle):
                pass
        total += outer
    return total


print(variable_inner(20000))
print(constant_inner(20000))
print(single_loop(20000))
print(three_levels(12000))
