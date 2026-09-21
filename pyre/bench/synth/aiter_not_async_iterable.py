# `aiter` on a non-async-iterable raises TypeError after an AttributeError
# miss on `type(obj).__aiter__`. compiles=0; the collecting exception
# materialisation is an interpreter rooting path.
# Expected: typeerr
try:
    aiter(object())
except TypeError:
    print("typeerr")
