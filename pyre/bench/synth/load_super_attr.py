N = 200000


class Base:
    def val(self):
        return 1


class Child(Base):
    def run(self, n):
        acc = 0
        i = 0
        while i < n:
            acc = acc + super(Child, self).val()
            i = i + 1
        return acc


print(Child().run(N))
