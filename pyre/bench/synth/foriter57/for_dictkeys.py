def f():
    d = {1: "a", 2: "b", 3: "c"}
    s = 0
    for k in d:
        s += k
    return s
print(f())
