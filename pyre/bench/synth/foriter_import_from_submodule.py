# pyre-check: max-pypy-ratio=70
# The `IMPORT_FROM` submodule fallback, inside a `for` body.
# `importing.rs import_from` falls through an AttributeError to a `sys.modules`
# lookup for `<__name__>.<name>`. `import_from_name_path` never reaches that
# lookup (its name is not a submodule either, so every iteration ends in
# ImportError) and `import_from_hot` takes the plain-getattr success in a
# `while` loop, which never consults the FOR_ITER body scan at all.
#
# Deleting the attribute forces the fallback. Nothing is stored back onto the
# parent, so `hasattr` stays False from the second iteration on and every
# iteration keeps taking it; `seen` counts the iterations that returned a
# module, so a dropped or doubled one changes the count. The printed `False` is
# the parity assertion: `__import__`'s `_handle_fromlist` returns the cached
# `encodings.utf_8` without rebinding the attribute, and this opcode must not
# rebind it either.
#
# `encodings` is the package every backend already has -- the codecs bootstrap
# imports `encodings.utf_8` before user code runs -- so this reaches wasm, which
# has no `os`/`posix`. wasm is also the backend that answers this from the
# native importer rather than `importlib._bootstrap`, so the printed `False` is
# what holds the two implementations to the same answer.
# Output verified against CPython/PyPy. The reading is 27x/33x and the ceiling
# is twice the slower of the two natives.
import encodings

N = 40000


def main():
    seen = 0
    for _ in range(N):
        if hasattr(encodings, "utf_8"):
            delattr(encodings, "utf_8")
        from encodings import utf_8
        if utf_8 is not None:
            seen += 1
    print(seen, hasattr(encodings, "utf_8"), utf_8.__name__)


main()
