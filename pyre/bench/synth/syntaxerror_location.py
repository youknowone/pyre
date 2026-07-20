# A compile-time SyntaxError carries its parser location: msg, lineno, offset,
# text, filename and end_lineno populate the instance (args is
# (msg, (filename, lineno, offset, text, end_lineno, end_offset))), and __str__
# renders 'msg (filename, line N)'.  The message text, offset and end positions
# differ between parsers, so this fixture asserts only the facts stable across
# CPython, PyPy and pyre's parser: attribute presence and types, the args
# shape, and the __str__ structure.
def main():
    try:
        compile("x = (1 +\n", "prog.py", "exec")
    except SyntaxError as e:
        print(type(e).__name__)
        print(e.lineno == 1)
        print(isinstance(e.offset, int) and e.offset >= 1)
        print(e.filename == "prog.py")
        print(isinstance(e.text, str) and "x = (1 +" in e.text)
        print(isinstance(e.end_lineno, int))
        print(len(e.args) == 2)
        print(e.args[0] == e.msg)
        print(isinstance(e.args[1], tuple) and len(e.args[1]) == 6)
        print(isinstance(e.msg, str) and str(e).startswith(e.msg))
        print("prog.py" in str(e) and "line" in str(e))


main()
