import _symtable
import sys


filenames = (b"\xff", "\udcff")
if sys.platform == "win32":
    # PEP 529 spells a filename as UTF-8 with `surrogatepass`, so the byte has
    # no spelling at all and the filename converter reports it before the
    # compiler is reached.  The surrogate keeps its own three-byte encoding and
    # still reaches the compiler as itself.
    filenames = ("\udcff",)

for filename in filenames:
    try:
        _symtable.symtable("x =", filename, "exec")
    except SyntaxError as exc:
        print(ascii(exc.filename))

print("OK")
