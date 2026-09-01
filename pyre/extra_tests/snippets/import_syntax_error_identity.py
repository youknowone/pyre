# pyre-check: gate=1
"""Source loaders retain SyntaxError identity and unlocated NUL refusals."""

import importlib.util
import os
import tempfile


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


with tempfile.TemporaryDirectory() as directory:
    invalid = os.path.join(directory, "invalid.py")
    with open(invalid, "wb") as stream:
        stream.write(b"if:\n")
    try:
        load(invalid, "review_invalid_syntax")
    except SyntaxError as exc:
        assert exc.filename == invalid
        assert exc.lineno == 1
    else:
        raise AssertionError("invalid imported source did not raise SyntaxError")

    nul = os.path.join(directory, "nul.py")
    with open(nul, "wb") as stream:
        stream.write(b"x\r\0")
    try:
        load(nul, "review_nul_source")
    except SyntaxError as exc:
        assert str(exc) == "source code string cannot contain null bytes"
        assert (exc.filename, exc.lineno, exc.offset, exc.text) == (None, None, None, None)
    else:
        raise AssertionError("NUL source was imported")

print("OK")
