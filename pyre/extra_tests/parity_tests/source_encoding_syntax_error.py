"""Malformed encoded source is the tokenizer's SyntaxError, not a decode error.

`pyparse.py:130-139 _handle_encoding` routes both an unknown codec and a codec
that rejects the bytes into the parser's SyntaxError; only an unrelated error
propagates.  The exception carries the file it came from, so `str()` renders the
`(name, line 0)` suffix.
"""

import os
import shutil
import sys
import tempfile

directory = tempfile.mkdtemp()
sys.path.insert(0, directory)


def write(name, payload):
    path = os.path.join(directory, name + ".py")
    with open(path, "wb") as stream:
        stream.write(payload)
    return path


write("enc_bad_bytes", b'# coding: ascii\nx = "\xff"\n')
write("enc_bad_codec", b"# coding: nosuchcodec\nx = 1\n")
write("enc_ok_latin1", b"# coding: latin-1\nx = '\xe9'\n")

try:
    try:
        import enc_bad_bytes
    except SyntaxError as error:
        assert error.msg == (
            "'ascii' codec can't decode byte 0xff in position 21: "
            "ordinal not in range(128)"
        ), error.msg
        assert os.path.basename(error.filename) == "enc_bad_bytes.py", error.filename
        assert error.lineno == 0, error.lineno
        assert error.text is None, error.text
        assert str(error).endswith("(enc_bad_bytes.py, line 0)"), str(error)
    else:
        raise AssertionError("undecodable source must not import")

    try:
        import enc_bad_codec
    except SyntaxError as error:
        assert error.msg == "unknown encoding: nosuchcodec", error.msg
        assert os.path.basename(error.filename) == "enc_bad_codec.py", error.filename
        assert error.lineno == 0, error.lineno
        assert str(error).endswith("(enc_bad_codec.py, line 0)"), str(error)
    else:
        raise AssertionError("an unknown codec must not import")

    # A declared codec that does decode is untouched by the conversion.
    import enc_ok_latin1

    assert enc_ok_latin1.x == "é", enc_ok_latin1.x

    # The same boundary backs `compile()` on bytes.
    try:
        compile(b'# coding: ascii\nx = "\xff"\n', "inline.py", "exec")
    except SyntaxError as error:
        assert "codec can't decode byte 0xff" in error.msg, error.msg
    else:
        raise AssertionError("compile() must reject undecodable bytes")
finally:
    sys.path.remove(directory)
    shutil.rmtree(directory, ignore_errors=True)

print("OK")
