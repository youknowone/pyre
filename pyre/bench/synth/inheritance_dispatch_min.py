N = 500000


class Base:
    def value(self):
        return 1


class Left(Base):
    def value(self):
        return 3


class Right(Base):
    def value(self):
        return 5


def main():
    objs = [Base(), Left(), Right()]
    i = 0
    acc = 0
    while i < N:
        acc = acc + objs[i % 3].value()
        i = i + 1
    print(acc)


main()
