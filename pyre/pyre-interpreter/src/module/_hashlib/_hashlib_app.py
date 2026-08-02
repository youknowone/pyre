"""App-level half of the _hashlib module.

`HASH` accumulates the fed data and re-computes the digest through the
interp-level `_hashlib._oneshot_digest` primitive on demand.  The
`openssl_<name>` constructors and `new` are interp-level (non-binding) and
build a `HASH`.
"""

_DIGEST_SIZE = {
    "md5": 16, "sha1": 20, "sha224": 28, "sha256": 32, "sha384": 48,
    "sha512": 64, "sha3_224": 28, "sha3_256": 32, "sha3_384": 48,
    "sha3_512": 64, "blake2b": 64, "blake2s": 32, "shake_128": 0,
    "shake_256": 0,
}

_BLOCK_SIZE = {
    "md5": 64, "sha1": 64, "sha224": 64, "sha256": 64, "sha384": 128,
    "sha512": 128, "sha3_224": 144, "sha3_256": 136, "sha3_384": 104,
    "sha3_512": 72, "blake2b": 128, "blake2s": 64, "shake_128": 168,
    "shake_256": 136,
}


class HASH:
    def __init__(self, name, data=b""):
        import _hashlib
        # The digest is named by the entry the name resolved to, so `name`,
        # `digest_size`, `block_size` and the digest itself all read the same
        # spelling however the caller spelled it.
        self._name = _hashlib._resolve_digest_name(name)
        self._data = bytearray()
        if data:
            self.update(data)

    @property
    def name(self):
        return self._name

    @property
    def digest_size(self):
        return _DIGEST_SIZE.get(self._name, 0)

    @property
    def block_size(self):
        return _BLOCK_SIZE.get(self._name, 64)

    def update(self, data):
        self._data += bytes(data)

    def _compute(self, length):
        import _hashlib
        return _hashlib._oneshot_digest(self._name, bytes(self._data), length)

    def digest(self, length=None):
        if self._name in ("shake_128", "shake_256"):
            if length is None:
                raise TypeError("digest() missing required argument 'length'")
            return self._compute(length)
        return self._compute(0)

    def hexdigest(self, length=None):
        return self.digest(length).hex()

    def copy(self):
        clone = HASH(self._name)
        clone._data = bytearray(self._data)
        return clone
