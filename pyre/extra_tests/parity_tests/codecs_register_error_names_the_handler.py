# CPython-suite gap: `test_codeccallbacks` registers handlers that work and
# checks the refusal for a non-callable by exception type only.
#
# parity-tests reason: `_codecs.register` and `PyCodec_RegisterError` each
# refuse a non-callable with their own noun -- `argument` for the search
# function the codec registry stores, `handler` for the error handler -- and
# both are reachable from the same `codecs` module, so one sentence for both
# tells a program the wrong thing about which registry it was calling.
import codecs


def refusal(fn):
    try:
        fn()
    except BaseException as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    raise AssertionError("accepted")


assert refusal(lambda: codecs.register(1)) == "TypeError: argument must be callable"
assert refusal(lambda: codecs.register_error("x", 1)) == (
    "TypeError: handler must be callable"
), refusal(lambda: codecs.register_error("x", 1))
assert refusal(lambda: codecs.register_error("x", None)) == "TypeError: handler must be callable"

# The name is read before the handler, so a bad name is refused by the argument
# rather than by the callable test below it.
assert "str" in refusal(lambda: codecs.register_error(1, len))

# A handler that is callable is accepted and reachable again by name.
def handler(exc):
    return ("?", exc.end)


codecs.register_error("parity-handler", handler)
assert codecs.lookup_error("parity-handler") is handler
assert "\xe9".encode("ascii", "parity-handler") == b"?"

print("OK")
