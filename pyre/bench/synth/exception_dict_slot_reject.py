# A subclass whose base already carries an instance dict rejects a second
# `__dict__` slot. BaseException subclasses carry the dict through the native
# exception slot, so they must reject it too.
def main():
    for base in (Exception, BaseException, ValueError):
        try:
            type("E", (base,), {"__slots__": ("__dict__",)})
            print(base.__name__, "+ slots(__dict__): no error")
        except TypeError as e:
            print(base.__name__, "+ slots(__dict__):", e)

    # object base has no instance dict, so the slot is allowed
    try:
        type("S", (object,), {"__slots__": ("__dict__",)})
        print("object + slots(__dict__): OK")
    except TypeError as e:
        print("object + slots(__dict__):", e)


main()
