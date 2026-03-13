n = 100000
a = 0
b = 1
i = 0
while i < n:
    t = a + b
    a = b
    b = t
    i = i + 1
print(b)
