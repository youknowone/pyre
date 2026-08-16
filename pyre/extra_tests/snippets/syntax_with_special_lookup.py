# pyre-check: gate=1
seen = [0, 0]
def enter(self):
    return self
def exit(self, typ, value, traceback):
    return False
class SpecialDescr:
    def __init__(self, impl, index):
        self.impl = impl
        self.index = index
    def __get__(self, obj, owner):
        seen[self.index] += 1
        return self.impl.__get__(obj, owner)
class Manager:
    __enter__ = SpecialDescr(enter, 0)
    __exit__ = SpecialDescr(exit, 1)
    def __getattribute__(self, name):
        raise AssertionError(name)
i = 0
while i < 3000:
    with Manager():
        pass
    i += 1

assert i == 3000
assert seen[0] == 3000
assert seen[1] == 3000
