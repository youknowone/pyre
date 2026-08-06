# pyre-check: max-pypy-ratio=7.8
# A __prepare__ namespace whose keys include a lone surrogate must be read back
# into the class without assuming the keys are valid UTF-8 (the class-statement
# / __build_class__ gather and the metaclass replay both iterate the keys).


class Meta(type):
    @classmethod
    def __prepare__(mcs, name, bases):
        # a lone-surrogate key sits alongside the ordinary ones
        return {chr(0xD800): 'surrogate', 'marker': 'ok'}


class C(metaclass=Meta):
    body_attr = 42


def main():
    # the ordinary prepared-namespace key survives the surrogate sibling
    print('marker', C.marker)
    # a normal class-body store still lands
    print('body', C.body_attr)
    # the surrogate-named attribute is preserved and retrievable
    print('surrogate', getattr(C, chr(0xD800)))


main()
