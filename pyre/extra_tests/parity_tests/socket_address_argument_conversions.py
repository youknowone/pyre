# CPython-suite gap: test_socket does not import, so nothing in the suite calls
# these conversions with an argument outside the range the C type holds.
# parity-tests reason: `htons` / `ntohs` / `htonl` / `ntohl` narrowed with a
# cast, so `htons(70000)` answered for port 4464 instead of refusing, and the
# name `<broadcast>` — which `inet_addr` cannot express, because failure and
# all-ones are the same bit pattern — went to the resolver and failed there.

# pyre-check: pypy-diverges: the `uint16_t` / `uint32_t` converter messages and
# `getservbyport`'s refusal are CPython 3.14's; pypy3 raises OverflowError with
# its own `c_uint` wording, and `socket.htons(70000)` answers 28689 there.

import socket

# The wildcard and the broadcast name never reach the resolver.
assert socket.gethostbyname("") == "0.0.0.0"
assert socket.gethostbyname("<broadcast>") == "255.255.255.255"
assert socket.gethostbyname("255.255.255.255") == "255.255.255.255"

# A socket address takes the same two names.
udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    udp.connect(("<broadcast>", 9))
    assert udp.getpeername() == ("255.255.255.255", 9), udp.getpeername()
finally:
    udp.close()

# The four byte-order conversions refuse what they cannot hold.
for convert, ctype in (
    (socket.htons, "uint16_t"),
    (socket.ntohs, "uint16_t"),
    (socket.htonl, "uint32_t"),
    (socket.ntohl, "uint32_t"),
):
    limit = 0xFFFF if ctype == "uint16_t" else 0xFFFFFFFF
    try:
        convert(limit + 1)
    except OverflowError as e:
        assert str(e) == "Python int too large for C " + ctype, str(e)
    else:
        raise SystemExit("%s accepted %d" % (convert.__name__, limit + 1))
    try:
        convert(-1)
    except ValueError as e:
        assert str(e) == "Cannot convert negative int", str(e)
    else:
        raise SystemExit("%s accepted -1" % convert.__name__)
    # The whole range still round-trips.
    assert convert(convert(limit)) == limit
    assert convert(0) == 0

# A port outside the range is refused rather than narrowed, and the message
# names the method that was called.
try:
    socket.socket().bind(("127.0.0.1", 70000))
except OverflowError as e:
    assert str(e) == "bind(): port must be 0-65535.", str(e)
else:
    raise SystemExit("bind accepted port 70000")

try:
    socket.getservbyport(70000)
except OverflowError as e:
    assert str(e) == "getservbyport: port must be 0-65535.", str(e)
else:
    raise SystemExit("getservbyport accepted port 70000")

# A family with no parser is a different failure from an address string the
# parser rejected: the first carries the errno the call left behind.
try:
    socket.inet_pton(1234, "1.2.3.4")
except OSError as e:
    assert e.errno is not None, e.args
else:
    raise SystemExit("inet_pton accepted family 1234")

try:
    socket.inet_pton(socket.AF_INET, "nope")
except OSError as e:
    assert e.errno is None, e.args
    assert str(e) == "illegal IP address string passed to inet_pton", str(e)
else:
    raise SystemExit("inet_pton accepted 'nope'")

print("OK")
