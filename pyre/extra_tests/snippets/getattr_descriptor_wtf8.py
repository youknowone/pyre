# pyre-check: gate=1
# DescrOperation.getattr/get and W_Property.get: builtin getattr follows the
# same descriptor protocol for UTF-8 and lone-surrogate names. In particular,
# the constant-name route must retain the receiver's method binding and the
# type-version, instance-shadowing, and property-fget invalidation guards.


def read_pair(obj, method_name, property_name):
    total = 0
    i = 0
    while i < 3000:
        total += getattr(obj, method_name)()
        total += getattr(obj, property_name)
        i += 1
    return total


def method(self):
    return 3


def replacement(self):
    return 7


def getter(self):
    return 5


def replacement_getter(self):
    return 11


class Example:
    pass


for method_name, property_name in [('method', 'value'), ('\udc81', '\udc82')]:
    descriptor = property(getter)
    setattr(Example, method_name, method)
    setattr(Example, property_name, descriptor)
    obj = Example()
    assert read_pair(obj, method_name, property_name) == 3000 * 8

    # A newly shadowing instance attribute must supersede the bound method.
    setattr(obj, method_name, lambda: 13)
    assert read_pair(obj, method_name, property_name) == 3000 * 18
    delattr(obj, method_name)

    setattr(Example, method_name, replacement)
    assert read_pair(obj, method_name, property_name) == 3000 * 12

    # Mutating the descriptor's fget does not replace the type-dict binding.
    descriptor.__init__(replacement_getter)
    assert read_pair(obj, method_name, property_name) == 3000 * 18
    assert getattr(obj, '\udcff_missing', 23) == 23


class Override(Example):
    def __getattribute__(self, name):
        if name == '\udc81':
            return lambda: 17
        return object.__getattribute__(self, name)


assert read_pair(Override(), '\udc81', '\udc82') == 3000 * 28
print('ok')
