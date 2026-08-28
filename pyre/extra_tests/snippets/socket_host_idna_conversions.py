# pyre-check: gate=1
# CPython-suite gap: test_socket does not import, so nothing in the suite runs
# a host argument through the two converters these entry points split between.
# parity-tests reason: the three `gethostby*` functions took the address-tuple
# converter, which hands an ASCII host straight through, and `getaddrinfo` took
# no codec at all, so a name the `idna` codec refuses reached the resolver and
# an internationalized name reached it spelled utf-8 rather than ACE.

# pyre-check: pypy-diverges: pins the 3.14 spelling of both converters; pypy3
# refuses a bytes host to `gethostbyname` with "'str' object expected, got
# 'bytes' instead", spells the wrong-type and embedded-null messages its own
# way, and lets the codec's own UnicodeError out of an address tuple.

import socket

IDNA_REFUSED = ("..bad..", "a" * 70 + ".example")

# The name lookups encode every `str` host with the `idna` codec, so a label
# that is empty or 64 bytes or longer is refused before the resolver sees it.
for name in IDNA_REFUSED:
    for call in (
        lambda h: socket.getaddrinfo(h, 80),
        socket.gethostbyname,
        socket.gethostbyname_ex,
        socket.gethostbyaddr,
    ):
        try:
            call(name)
        except UnicodeEncodeError as e:
            assert e.encoding == "idna", e.encoding
            assert e.object == name, e.object
        else:
            raise SystemExit("%r was accepted" % (name,))

# A bytes host already names an encoded host and is passed through untouched.
assert socket.gethostbyname(b"127.0.0.1") == "127.0.0.1"
assert socket.gethostbyname(bytearray(b"127.0.0.1")) == "127.0.0.1"

# The wrong type and an embedded null are both TypeErrors naming the function
# the argument was passed to.
for bad, spelled in ((42, "int"), (None, "None"), (memoryview(b"x"), "memoryview")):
    try:
        socket.gethostbyname(bad)
    except TypeError as e:
        assert str(e) == (
            "gethostbyname() argument 1 must be str, bytes or bytearray, not " + spelled
        ), str(e)
    else:
        raise SystemExit("gethostbyname accepted %r" % (bad,))

for null_host, spelled in (("127.0.0.1\x00zz", "str"), (b"127.0.0.1\x00zz", "bytes")):
    try:
        socket.gethostbyname(null_host)
    except TypeError as e:
        assert str(e) == (
            "gethostbyname() argument 1 must be encoded string without null bytes,"
            " not " + spelled
        ), str(e)
    else:
        raise SystemExit("gethostbyname accepted an embedded null")

# `getaddrinfo` hands its host and its service to the resolver as C strings, so
# each simply stops at the first null.
numeric = socket.getaddrinfo("127.0.0.1", 80, socket.AF_INET)
assert socket.getaddrinfo("127.0.0.1\x00zz", 80, socket.AF_INET) == numeric
assert socket.getaddrinfo(b"127.0.0.1\x00zz", 80, socket.AF_INET) == numeric
assert socket.getaddrinfo("127.0.0.1", "80\x00zz", socket.AF_INET) == numeric

# An address tuple takes the other converter: an ASCII `str` never enters the
# codec, and bytes and bytearray are accepted beside it.
for host in ("127.0.0.1", b"127.0.0.1", bytearray(b"127.0.0.1")):
    sock = socket.socket()
    try:
        sock.bind((host, 0))
    finally:
        sock.close()

for bad, spelled in ((42, "int"), (None, "NoneType"), (memoryview(b"x"), "memoryview")):
    sock = socket.socket()
    try:
        sock.bind((bad, 0))
    except TypeError as e:
        assert str(e) == "str, bytes or bytearray expected, not " + spelled, str(e)
    else:
        raise SystemExit("bind accepted %r" % (bad,))
    finally:
        sock.close()

sock = socket.socket()
try:
    sock.bind(("127.0.0.1\x00zz", 0))
except TypeError as e:
    assert str(e) == "host name must not contain null character", str(e)
else:
    raise SystemExit("bind accepted an embedded null")
finally:
    sock.close()


# The fast path is the compact-ASCII representation, which no `str` subclass
# instance has, so a subclass host enters the codec, and a codec failure there
# is reported as one message rather than as the codec's own error.
class Host(str):
    pass


sock = socket.socket()
try:
    sock.bind((Host("a" * 70 + ".example"), 0))
except TypeError as e:
    assert str(e) == "encoding of hostname failed", str(e)
else:
    raise SystemExit("bind accepted a subclass host the codec refuses")
finally:
    sock.close()

print("OK")
