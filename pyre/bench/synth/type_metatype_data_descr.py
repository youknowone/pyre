# pyre-check: max-pypy-ratio=14
# pypy's exec time is pinned to the startup-subtraction floor here, so the
# ratio is not a measurement: the ceiling is twice the slowest ratio the CI
# runners observe (6.4x), rounded up, and no floor gate is recorded.
# typeobject.py W_TypeObject.descr_getattribute: a metatype DATA descriptor
# wins over the class's own MRO value of the same name; a metatype non-data
# descriptor and a plain metatype attribute lose to the class's own value.


class DataDesc:
    def __get__(self, obj, objtype=None):
        return 'meta-data'

    def __set__(self, obj, value):
        pass


class NonDataDesc:
    def __get__(self, obj, objtype=None):
        return 'meta-nondata'


class Meta(type):
    data = DataDesc()
    nondata = NonDataDesc()
    plain = 'meta-plain'


class C(metaclass=Meta):
    data = 'own-data'
    nondata = 'own-nondata'
    plain = 'own-plain'
    only_own = 'own-only'


def main():
    # metatype data descriptor beats the class's own value
    print('data', C.data)
    # class's own value beats a metatype non-data descriptor
    print('nondata', C.nondata)
    # class's own value beats a plain metatype attribute
    print('plain', C.plain)
    # class-only attribute still resolves
    print('only_own', C.only_own)


main()
