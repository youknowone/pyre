"""App-level _blake2 constructors backed by the _hashlib HASH object.

Only the default digest sizes (blake2b-512 / blake2s-256) are computed;
the keyed/salted/personalised parameters are accepted and ignored.
"""


def blake2b(data=b"", *, digest_size=64, key=b"", salt=b"", person=b"",
            fanout=1, depth=1, leaf_size=0, node_offset=0, node_depth=0,
            inner_size=0, last_node=False, usedforsecurity=True):
    from _hashlib import HASH
    return HASH("blake2b", data)


def blake2s(data=b"", *, digest_size=32, key=b"", salt=b"", person=b"",
            fanout=1, depth=1, leaf_size=0, node_offset=0, node_depth=0,
            inner_size=0, last_node=False, usedforsecurity=True):
    from _hashlib import HASH
    return HASH("blake2s", data)
