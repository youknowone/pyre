# pyre-check: gate=1
"""PyPy's `_codecs_cn` cjkcodecs engine, including persistent HZ state."""

import _codecs_cn
import codecs
import gc
import sys
from _multibytecodec import MultibyteIncrementalDecoder, MultibyteIncrementalEncoder


assert MultibyteIncrementalDecoder.__base__ is object
assert MultibyteIncrementalEncoder.__base__ is object


def oversized_position(exc):
    replacement = "" if isinstance(exc, UnicodeDecodeError) else "?"
    return replacement, sys.maxsize + 1


codecs.register_error("cn_oversized_position", oversized_position)
for operation in (
    lambda: b"\xff".decode("gbk", "cn_oversized_position"),
    lambda: "\u2603".encode("gbk", "cn_oversized_position"),
):
    try:
        operation()
    except IndexError:
        pass
    else:
        raise AssertionError("an oversized error-handler position must be rejected")


for name, text in (
    ("gb2312", "你好"),
    ("gbk", "你好·—"),
    ("gb18030", "你好\U00012345"),
    ("hz", "聊聊~A"),
):
    codec = _codecs_cn.getcodec(name)
    encoded, consumed = codec.encode(text)
    assert consumed == len(text), (name, consumed)
    assert codec.decode(encoded) == (text, len(encoded)), name


for name, text, encoded in (
    ("gb2312", "・", bytes.fromhex("a1a4")),
    ("gbk", "·—―", bytes.fromhex("a1a4a1aaa844")),
    ("gb18030", "\x80・€\U00010000😀", bytes.fromhex("813081308139a739a2e3903081309439fc36")),
):
    codec = _codecs_cn.getcodec(name)
    assert codec.encode(text) == (encoded, len(text)), name
    assert codec.decode(encoded) == (text, len(encoded)), name


# PyPy keeps this shift mode on one persistent encodebuf/decodebuf.  The
# app-level port serializes the same MultibyteCodec_State across calls; the
# exposed integers follow CPython 3.14's endian-independent codec contract.
encoder = codecs.getincrementalencoder("hz")()
try:
    del encoder.errors
except AttributeError as exc:
    assert str(exc) == "cannot delete attribute"
else:
    raise AssertionError("the errors descriptor must not be deletable")
try:
    encoder.errors = None
except TypeError as exc:
    assert str(exc) == "errors must be a string"
else:
    raise AssertionError("the errors descriptor must accept strings only")
encoder.errors = "strict"
assert encoder.getstate() == 0
assert encoder.encode("聊") == b"~{AD"
gb_state = encoder.getstate()
assert gb_state == 256, gb_state
assert encoder.encode("聊") == b"AD"
encoder.setstate(gb_state)
assert encoder.encode("聊") == b"AD"
assert encoder.encode("", final=True) == b"~}"
assert encoder.getstate() == 0

decoder = codecs.getincrementaldecoder("hz")()
assert decoder.getstate() == (b"", 0)
assert codecs.getincrementaldecoder("gbk")().decode(bytearray(b"abc")) == "abc"
assert codecs.getincrementaldecoder("gbk")().decode(memoryview(b"abc")) == "abc"
assert decoder.decode(b"~{") == ""
gb_state = decoder.getstate()
assert gb_state == (b"", 1), gb_state
assert decoder.decode(b"AD") == "聊"
decoder.setstate(gb_state)
assert decoder.decode(b"AD") == "聊"
assert decoder.decode(b"~}", final=True) == ""
assert decoder.getstate() == (b"", 0)


class ErrorName(str):
    def __str__(self):
        raise AssertionError("codec error names are extracted without __str__")

    def __getitem__(self, key):
        raise AssertionError("codec error names are extracted without overrides")


assigned_errors = ErrorName("ignore")
decoder.errors = assigned_errors
assert decoder.errors == "ignore"
assert type(decoder.errors) is str
assert decoder.errors is not assigned_errors


class CodecState(int):
    def to_bytes(self, *args, **kwargs):
        raise AssertionError("codec setstate must bypass int.to_bytes overrides")


codecs.getincrementaldecoder("gbk")().setstate((b"", CodecState(0)))
codecs.getincrementalencoder("gbk")().setstate(CodecState(0))

# The persistent PyPy engine is not transactional: a shift completed before a
# later bad code point/byte remains live after the exception.
encoder = codecs.getincrementalencoder("hz")()
try:
    encoder.encode("聊\udcff")
except UnicodeEncodeError:
    pass
else:
    raise AssertionError("HZ should reject a lone surrogate")
assert encoder.getstate() == 256
assert encoder.encode("聊") == b"AD"


# A text replacement is encoded by a nested engine with the outer engine's
# state.  PyPy's `c_codecs.encode(copystate=encodebuf)` copies that nested
# state back from its finally arm, including when the replacement itself
# raises.  CPython 3.14 exposes the resulting GB shift through getstate().
def nested_bad_replacement(exc):
    # Move young objects while the Rust/C codec engine is suspended in a
    # Python callback; its object-owned state bytearray must remain rooted.
    gc.collect()
    return "聊\udcff", exc.end


codecs.register_error("hz_nested_bad_replacement", nested_bad_replacement)
encoder = codecs.getincrementalencoder("hz")("hz_nested_bad_replacement")
try:
    encoder.encode("\udcff")
except UnicodeEncodeError:
    pass
else:
    raise AssertionError("the nested HZ replacement should reject its surrogate")
assert encoder.getstate() == 256
assert encoder.encode("聊") == b"AD"

decoder = codecs.getincrementaldecoder("hz")()
try:
    decoder.decode(b"~{yy")
except UnicodeDecodeError:
    pass
else:
    raise AssertionError("HZ should reject an unmapped GB pair")
assert decoder.getstate() == (b"", 1)
assert decoder.decode(b"AD") == "聊"


def collecting_decode_ignore(exc):
    gc.collect()
    return "", exc.end


codecs.register_error("hz_collecting_decode_ignore", collecting_decode_ignore)
decoder = codecs.getincrementaldecoder("hz")("hz_collecting_decode_ignore")
assert decoder.decode(b"~{yy") == ""
assert decoder.getstate() == (b"y", 1)

print("OK")
