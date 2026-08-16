# pyre-check: gate=1
class Base:
    pass
class Meta(Base, type):
    pass
class C(metaclass=Meta):
    pass
type_set_rejected = False
try:
    Base.__dict__['__dict__'].__set__(C, {})
except (TypeError, AttributeError):
    type_set_rejected = True
type_mapping_rejected = False
try:
    C.__dict__['x'] = 1
except TypeError:
    type_mapping_rejected = True
class Exc(Base, Exception):
    pass
exception_delete_rejected = False
try:
    Base.__dict__['__dict__'].__delete__(Exc())
except (TypeError, AttributeError):
    exception_delete_rejected = True

assert type_set_rejected
assert type_mapping_rejected
assert exception_delete_rejected
