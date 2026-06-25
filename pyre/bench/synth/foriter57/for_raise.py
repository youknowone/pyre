def f():
    total = 0
    try:
        for x in range(10):
            if x == 5:
                raise ValueError("stop")
            total += x
    except ValueError:
        total += 1000
    return total
print(f())
