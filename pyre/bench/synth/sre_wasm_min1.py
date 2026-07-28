import re

N = 2500
pair = re.compile(r"([a-z]+)(\d+)")


def main():
    subject = "ab12 cd34 ef56"
    acc = 0
    i = 0
    while i < N:
        for m in pair.finditer(subject):
            acc = acc + len(m.expand(r"\1"))
        i = i + 1
    print(acc)


main()
