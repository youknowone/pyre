# A failed `from X import Y` raises ImportError carrying `.name` (the module)
# and `.path` (its source file), per error.py new_import_error. The absolute
# path is install-dependent, so only its basename is checked.  `keyword` is a
# pure-Python stdlib module importable on every backend (including wasm, which
# has no `os`/`posix`).  Output verified against CPython/PyPy.
N = 4000


def main():
    ok = 0
    for _ in range(N):
        try:
            from keyword import _no_such_name_xyz_
        except ImportError as e:
            if (
                type(e).__name__ == "ImportError"
                and e.name == "keyword"
                and isinstance(e.path, str)
                and e.path.endswith("keyword.py")
            ):
                ok += 1
    print(ok)


main()
