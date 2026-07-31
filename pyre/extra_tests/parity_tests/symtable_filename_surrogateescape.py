import _symtable
import sys


filenames = (b"\xff", "\udcff")
if sys.implementation.name == "cpython" and sys.platform == "win32":
    # CPython's Windows filesystem codec rejects both surrogateescaped
    # spellings before the compiler can preserve the filename. PyPy/pyre use
    # the byte-preserving route on every platform.
    filenames = ()

for filename in filenames:
    try:
        _symtable.symtable("x =", filename, "exec")
    except SyntaxError as exc:
        print(ascii(exc.filename))

print("OK")
