# complex.real / complex.imag are read-only getset descriptors bound to the
# complex type: they carry __objclass__ == complex and __name__, reject a
# non-complex receiver with TypeError, and reject assignment with
# AttributeError. (The descriptor type name and the exact TypeError/AttributeError
# text are not printed here: they diverge between the reference interpreters.)
def main():
    c = complex(3, -4)
    print(c.real, c.imag)
    print(complex.real.__get__(c), complex.imag.__get__(c))
    print(complex.real.__objclass__.__name__, complex.imag.__objclass__.__name__)
    print(complex.real.__name__, complex.imag.__name__)
    for bad in ["x", 42, 3.0]:
        try:
            complex.real.__get__(bad)
            print("no raise")
        except TypeError:
            print("TypeError")
    try:
        c.real = 1
        print("no raise")
    except AttributeError:
        print("AttributeError")


main()
