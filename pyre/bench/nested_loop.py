def main():
    s = 0
    i = 0
    while i < 20000:
        j = 0
        while j < 20000:
            s = s + i * j
            j = j + 1
        i = i + 1
    print(s)

main()
