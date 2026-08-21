# pyre-check: gate=1
"""A codec lookup runs Python, so the object being decoded must survive it.

`codecs.register` search functions run while the decode is already under way:
an uncached name imports its `encodings` module and calls every registered
search function before the codec's own `decode` is reached.  The object handed
to that decode is a copy the interpreter made of the source bytes, so it has to
stay rooted across the lookup rather than only across the final call.
"""
import codecs
import gc

SRC = bytes(range(128))
TEXT = "".join(chr(i) for i in range(128))
seen = []


def probe_decode(b, errors="strict"):
    raw = bytes(b)
    seen.append(raw)
    return (raw.decode("latin-1"), len(raw))


def probe_encode(u, errors="strict"):
    seen.append(u)
    return (u.encode("latin-1"), len(u))


def make_search(name):
    def search(query):
        if query != name:
            return None
        for _ in range(3):
            gc.collect()
        return codecs.CodecInfo(name=name, encode=probe_encode, decode=probe_decode)

    return search


codecs.register(make_search("pyreprobedec"))
codecs.register(make_search("pyreprobeenc"))

assert str(SRC, "pyreprobedec") == TEXT
assert seen[-1] == SRC, seen[-1]

assert TEXT.encode("pyreprobeenc") == SRC
assert seen[-1] == TEXT, seen[-1]
