class ClassArgs(list):
    ARGS = (1, 2)

    @classmethod
    def __getnewargs__(cls):
        return cls.ARGS


class StaticArgs(list):
    @staticmethod
    def __getnewargs__():
        return (3, 4)


class ExtendedArgs(int):
    ARGS = (5,)
    KWARGS = {"value": 6}

    def __new__(cls, marker, *, value):
        obj = super().__new__(cls, value)
        obj.marker = marker
        return obj

    @classmethod
    def __getnewargs_ex__(cls):
        return (cls.ARGS, cls.KWARGS)


class BindingDescriptor:
    def __get__(self, obj, cls):
        assert type(obj) is DescriptorArgs
        assert cls is DescriptorArgs
        return lambda: (7, 8)


class DescriptorArgs(list):
    __getnewargs__ = BindingDescriptor()


assert ClassArgs().__reduce_ex__(2)[1][1:] == (1, 2)
assert StaticArgs().__reduce_ex__(2)[1][1:] == (3, 4)

extended_reduce_args = ExtendedArgs(5, value=6).__reduce_ex__(2)[1]
assert extended_reduce_args[1] == (5,)
assert extended_reduce_args[2] == {"value": 6}

assert DescriptorArgs().__reduce_ex__(2)[1][1:] == (7, 8)

print("OK")
