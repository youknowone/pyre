# pyre-check: max-pypy-ratio=5
# util.py:80 _objectdir / objectobject.py:324 — dir() of a dict instance lists
# the dict type's attributes (object.__dir__), NOT the dict's own keys.


def main():
    d = {'zzz_key': 1, 'aaa_key': 2}
    names = dir(d)
    # the keys are NOT attribute names
    print('zzz_key in dir', 'zzz_key' in names)
    print('aaa_key in dir', 'aaa_key' in names)
    # dict methods ARE
    print('keys in dir', 'keys' in names)
    print('__contains__ in dir', '__contains__' in names)
    print('__setitem__ in dir', '__setitem__' in names)


main()
