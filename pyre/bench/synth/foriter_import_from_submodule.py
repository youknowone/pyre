# pyre-check: max-pypy-ratio=120
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
# Output verified against CPython/PyPy. The reading moved over 29x-33x-41x
# across three runs -- this is startup-dominated and noisy -- so the ceiling is
# three times the slowest of them, the headroom `import_from_hot` carries for
# the same reason.
import encodings
import sys
import types

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


# IMPORT_FROM's fallback reads `<parent.__name__>.<child>` from sys.modules;
# the parent need not itself be a module, and the opcode must not bind the
# answer back onto it.
class Stand:
    __slots__ = ("__name__",)


stand = Stand()
stand.__name__ = "pyre_stand"
piece = types.ModuleType("pyre_stand.piece")
sys.modules["pyre_stand"] = stand
sys.modules["pyre_stand.piece"] = piece
from pyre_stand import piece as got_piece
assert got_piece is piece
assert not hasattr(stand, "piece")
