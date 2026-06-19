"""App-level half of the _hashlib module.

`HASH` accumulates the fed data and re-computes the digest through the
interp-level `_hashlib._oneshot_digest` primitive on demand.  The
`openssl_<name>` constructors and `new` mirror the OpenSSL surface
`hashlib.py` expects.
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
        self._name = name
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


def new(name, string=b"", *, usedforsecurity=True, **kwargs):
    return HASH(name, string)


def _ctor(_name):
    def constructor(string=b"", *, usedforsecurity=True, **kwargs):
        return HASH(_name, string)
    constructor.__name__ = "openssl_" + _name
    return constructor


openssl_md5 = _ctor("md5")
openssl_sha1 = _ctor("sha1")
openssl_sha224 = _ctor("sha224")
openssl_sha256 = _ctor("sha256")
openssl_sha384 = _ctor("sha384")
openssl_sha512 = _ctor("sha512")
openssl_sha3_224 = _ctor("sha3_224")
openssl_sha3_256 = _ctor("sha3_256")
openssl_sha3_384 = _ctor("sha3_384")
openssl_sha3_512 = _ctor("sha3_512")
openssl_shake_128 = _ctor("shake_128")
openssl_shake_256 = _ctor("shake_256")
openssl_blake2b = _ctor("blake2b")
openssl_blake2s = _ctor("blake2s")
