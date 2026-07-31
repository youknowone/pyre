# A metaclass resolves `Cls.name()` before the class's own MRO does:
# `type.__getattribute__` lets a metatype DATA descriptor win outright, and a
# metatype `__getattribute__` override produces the value itself. Either way the
# call must use what the metaclass returned, not rebind the class onto it.


class MetaProp(type):
    @property
    def where(cls):
        return lambda: 'meta-prop'


class ByProp(metaclass=MetaProp):
    @classmethod
    def where(cls):
        return 'own-classmethod'


class MetaGetattr(type):
    def __getattribute__(cls, name):
        if name == 'ping':
            return lambda: 'meta-getattr'
        return type.__getattribute__(cls, name)


class ByGetattr(metaclass=MetaGetattr):
    @classmethod
    def ping(cls):
        return 'own-classmethod'


class Plain:
    @classmethod
    def tag(cls):
        return cls.__name__


def main():
    prop = getattr_ = plain = None
    for _ in range(20000):
        prop = ByProp.where()
        getattr_ = ByGetattr.ping()
        # an ordinary class still binds its classmethod's cls
        plain = Plain.tag()
    print('prop', prop)
    print('getattr', getattr_)
    print('plain', plain)


main()
