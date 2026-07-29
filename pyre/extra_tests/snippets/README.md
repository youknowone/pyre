# snippets fixture file

The snippets run with this directory as their working directory, and a
handful of them (`builtin_open.py`, `stdlib_io.py`, `stdlib_os.py`,
`stdlib_socket.py`) open `README.md` as a convenient read-only file — they
were imported from RustPython, whose runner started at the repository root.
This file stands in for that one, so the imported sources stay verbatim.

`builtin_open.py` asserts the text `RustPython` appears here.
