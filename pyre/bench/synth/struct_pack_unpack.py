import struct


def main():
    s = struct.Struct("<i")
    acc = 0
    i = 0
    while i < 200000:
        data = s.pack(i)
        (v,) = s.unpack(data)
        acc = acc + v
        i = i + 1
    print(acc)


main()
