# CPython-suite gap: function-attribute tests do not build definitions inside
# a loop hot enough to trace their SET_FUNCTION_ATTRIBUTE stores.
# parity-tests reason: this keeps only the object-identity and mutation checks
# not owned by `bench/synth/make_function_inline.py`.

"""Hot MAKE_FUNCTION retains the objects stamped onto each fresh function."""

N = 3000


def defaults_identity_loop():
    for i in range(N):
        marker = (i, i + 1)

        def take(value=marker):
            return value

        defaults = take.__defaults__
        assert defaults is not None and len(defaults) == 1
        assert defaults[0] is marker
        assert take() is marker


def kwdefaults_mutation_loop():
    total = 0
    for i in range(N):
        def take(*, value=i):
            return value

        defaults = take.__kwdefaults__
        assert defaults == {"value": i}
        defaults["value"] = i + 100
        total += take()
    assert total == sum(i + 100 for i in range(N))


def annotated_default_loop():
    for i in range(N):
        marker = (i,)

        def take(value: tuple = marker) -> tuple:
            return value

        assert take.__annotations__ == {"value": tuple, "return": tuple}
        assert take.__defaults__[0] is marker
        assert take() is marker


defaults_identity_loop()
kwdefaults_mutation_loop()
annotated_default_loop()
print("OK")
