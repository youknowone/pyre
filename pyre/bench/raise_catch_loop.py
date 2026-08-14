# pyre-check: max-wasm-ratio=3.8
# Reported 3.2x and 3.3x on ubuntu-24.04.

def main():
    s = 0
    i = 0
    while i < 100000000:
        try:
            if i % 7 == 0:
                raise ValueError("v")
            s = s + 1
        except ValueError:
            s = s + 2
        i = i + 1
    print(s)

main()
