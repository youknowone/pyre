string_ops__N = 125000

def string_ops__main():
    words = ['alpha', 'beta', 'gamma', 'delta']
    i = 0
    acc = 0
    while i < string_ops__N:
        s = words[i & 3]
        t = s + ':' + str(i & 255)
        if t.startswith('a') or t.endswith('7'):
            acc = acc + len(t)
        else:
            acc = acc - len(s)
        i = i + 1
    print(acc)
string_ops__main()

