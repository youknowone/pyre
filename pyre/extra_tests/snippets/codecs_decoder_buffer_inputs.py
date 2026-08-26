# pyre-check: gate=1
# Every named decoder unwraps its input the same way: any buffer is read, and
# a view whose bytes are strided is refused because they are not the sequence
# the object exposes.  The named codecs used to reach the object's own
# `decode`, which only `bytes` and `bytearray` have, so a `memoryview` was
# reported as the wrong argument type instead of being read.

import codecs

DECODERS = [
    codecs.ascii_decode,
    codecs.latin_1_decode,
    codecs.utf_8_decode,
]

for decode in DECODERS:
    assert decode(b"abc") == ("abc", 3), decode
    assert decode(bytearray(b"abc")) == ("abc", 3), decode
    assert decode(memoryview(b"abc")) == ("abc", 3), decode

assert codecs.charmap_decode(memoryview(b"abc"), "strict", None) == ("abc", 3)

# A strided view says so with the error the acquisition raised, not with the
# wording that names a wrong argument type.
strided = memoryview(b"abcdef")[::2]
for decode in DECODERS:
    try:
        decode(strided)
    except BufferError as error:
        assert "contiguous" in str(error), str(error)
    else:
        raise AssertionError("a strided view should not be readable: %r" % decode)

# Something that is no buffer at all is still the wrong argument type.
for decode in DECODERS:
    try:
        decode(1)
    except TypeError as error:
        assert "bytes-like object is required" in str(error), str(error)
    else:
        raise AssertionError("an int should not be readable: %r" % decode)

print("OK")
