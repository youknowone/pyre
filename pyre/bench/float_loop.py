def main():
    s = 0.0
    i = 0
    while i < 200000000:
        s = s + i * 0.1
        i = i + 1
    print(s)

main()
