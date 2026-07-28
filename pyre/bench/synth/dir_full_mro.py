# pyre-check: max-pypy-ratio=5
# util.py:62 _classdir / :80 _objectdir — dir() unions the __dict__ keys of a
# class and ALL of its bases recursively (the full MRO), and for an instance it
# adds the instance __dict__ on top.


class A:
    def a_meth(self):
        pass


class B(A):
    def b_meth(self):
        pass


class C(B):
    def c_meth(self):
        pass


def main():
    c = C()
    c.inst_attr = 1
    names = dir(c)
    # methods from every class in the MRO are present
    print('a_meth', 'a_meth' in names)
    print('b_meth', 'b_meth' in names)
    print('c_meth', 'c_meth' in names)
    # instance dict entry too
    print('inst_attr', 'inst_attr' in names)
    # dir(type) walks the full MRO as well
    tnames = dir(C)
    print('type_a_meth', 'a_meth' in tnames)
    print('type_b_meth', 'b_meth' in tnames)


main()
