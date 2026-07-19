# SyntaxError.__str__ renders 'msg (filename, line lineno)', degrading through
# the filename-only, lineno-only and bare-msg shapes; a non-str msg falls back
# to str(msg). (The end_lineno range form is not exercised here: PyPy renders
# 'lines N-M' while CPython keeps 'line N', so the two references diverge.)
def main():
    cases = [
        SyntaxError("bad", ("f.py", 3, 5, "code")),
        SyntaxError("bad", ("/a/b/f.py", 3, 5, "code")),
        SyntaxError("bad"),
        SyntaxError(),
        SyntaxError("only msg", (None, 7, 1, "x")),
    ]
    for e in cases:
        print(repr(str(e)))


main()
