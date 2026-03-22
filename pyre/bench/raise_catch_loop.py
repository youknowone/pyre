def main():
    s = 0
    i = 0
    while i < 1000000:
        try:
            if i % 7 == 0:
                raise ValueError("v")
            s = s + 1
        except ValueError:
            s = s + 2
        i = i + 1
    print(s)

main()
