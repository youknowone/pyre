bytes_ops__N = 175000

def bytes_ops__main():
    data = b'abcdefghijklmnopqrstuvwxyz'
    i = 0
    acc = 0
    while i < bytes_ops__N:
        b = data[i % len(data)]
        piece = data[i & 7:(i & 7) + 5]
        acc = acc + b + len(piece)
        i = i + 1
    print(acc)
bytes_ops__main()
