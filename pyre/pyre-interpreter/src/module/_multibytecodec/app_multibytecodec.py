"""PyPy-compatible app-level surface over the cjkcodecs engine."""


# `PyArg_ParseTupleAndKeywords` leaves an argument the caller omitted at the C
# NULL its variable was initialized to, and `internal_error_callback(NULL)`
# reads that as "strict".  An explicit `None` reaches the `s` converter
# instead, which rejects it.  A sentinel is what keeps the two apart here.
_OMITTED = object()


def _codec_errors_arg(name, errors, _str_getitem=str.__getitem__, _whole=slice(None)):
    # The `errors: str(accept={str, NoneType})` the two `MultibyteCodec`
    # methods declare: `None` is the "strict" spelling.
    if errors is None:
        return "strict"
    if not isinstance(errors, str):
        raise TypeError(
            f"{name}() argument 'errors' must be str or None, "
            f"not {type(errors).__name__}"
        )
    # `text_or_none` is a gateway conversion in PyPy, so rebinding names in
    # the app module cannot change it.  Capture the two builtin operands in
    # defaults instead of resolving `str` or `slice` through mutable globals.
    return _str_getitem(errors, _whole)


def _errors_arg(name, position, errors, _str_getitem=str.__getitem__, _whole=slice(None)):
    # The plain `s` the four initializers declare: only a str, and an omitted
    # one means "strict".
    if errors is _OMITTED:
        return "strict"
    if not isinstance(errors, str):
        # `seterror` names `None` itself rather than its type.
        got = "None" if errors is None else type(errors).__name__
        raise TypeError(f"{name}() argument {position} must be str, not {got}")
    return _str_getitem(errors, _whole)


def _codec_of(obj):
    # `mbiencoder_init` and its three siblings read `codec` off the type, not
    # off the instance, so the four base classes -- which carry none -- raise
    # right here rather than at the first encode or decode.
    codec = type(obj).codec
    if not isinstance(codec, MultibyteCodec):
        raise TypeError("codec is unexpected type")
    return codec


def _bufferstr_bytes(object, _memoryview=memoryview):
    # PyPy's `bufferstr_w` acquires one C-contiguous read-only byte view.  A
    # multi-byte-element exporter is accepted and its consumed position is
    # still measured in bytes, but a strided view must not be flattened into a
    # different byte sequence by `memoryview.tobytes()`.
    view = _memoryview(object)
    if not view.c_contiguous:
        raise BufferError("memoryview: underlying buffer is not C-contiguous")
    return view.tobytes()


class MultibyteCodec:
    def __init__(self, name):
        self.name = name

    def encode(self, input, errors=None):
        return _encode(self.name, input, _codec_errors_arg("encode", errors), True)

    def decode(self, input, errors=None):
        return _decode(self.name, input, _codec_errors_arg("decode", errors), True)


def _get_errors(self):
    return self._errors


def _set_errors(self, value, _str_getitem=str.__getitem__, _whole=slice(None)):
    if not isinstance(value, str):
        raise TypeError("errors must be a string")
    # PyPy `fset_errors` stores `space.text_w(w_errors)` and `fget_errors`
    # mints a fresh text object, so a str subclass never becomes the stored
    # observable value.  Call the base implementation directly to bypass a
    # subclass's `__getitem__` override while normalizing it to exact `str`.
    self._errors = _str_getitem(value, _whole)


def _del_errors(self):
    raise AttributeError("cannot delete attribute")


# PyPy `MultibyteIncrementalBase.errors` is a GetSetProperty backed by
# `fget_errors`/`fset_errors`, so it cannot be deleted and every write is
# text-checked.  Sharing the descriptor preserves that behavior without making
# the internal PyPy base an observable Python `__base__`; both exported types
# have `object` there in the real pypy3 and pinned 3.14.  The messages are the
# pinned 3.14 observable spellings.
_errors_property = property(_get_errors, _set_errors, _del_errors)


class MultibyteIncrementalDecoder:
    errors = _errors_property
    # The name and argument position the initializer reports a non-str
    # `errors` under; `MultibyteStreamReader` reuses the body under its own.
    _init_name = "IncrementalDecoder"
    _init_errors_position = 1

    def __init__(self, errors=_OMITTED):
        cls = type(self)
        self._errors = _errors_arg(cls._init_name, cls._init_errors_position, errors)
        self.codec = _codec_of(self)
        self.pending = b""
        # PyPy's `MultibyteIncrementalDecoder._initialize` owns one persistent
        # `decodebuf`; these bytes are its `MultibyteCodec_State.c` while the
        # app-level port crosses the Rust engine boundary one call at a time.
        self.state = bytearray(_initial_state(self.codec.name, True))

    def decode(self, object, final=False):
        # PyPy `decode_w(object='bufferstr')` hands the RPython codec a byte
        # string.  `consumed` is consequently a byte offset even when the
        # caller supplied a multi-byte-element buffer such as array('H').
        # Materialize before both dispatch and the pending suffix slice.
        data = self.pending + _bufferstr_bytes(object)
        output, consumed = _decode_stateful(
            self.codec.name, data, self.errors, (final, self.state)
        )
        self.pending = memoryview(data)[consumed:].tobytes()
        return output

    def reset(self):
        self.pending = b""
        self.state = bytearray(_initial_state(self.codec.name, True))

    def getstate(self):
        # Pinned v3.14.6
        # `Modules/cjkcodecs/multibytecodec.c::_multibytecodec_MultibyteIncrementalDecoder_getstate_impl`
        # exposes the decoder's native `state.c` as the second tuple item.
        # PyPy `MultibyteIncrementalDecoder.getstate_w` returns its separate
        # integer field instead.  Keep PyPy's persistent decodebuf engineering;
        # only this observable integer boundary is a 3.14-spec adaptation.
        return self.pending, int.from_bytes(self.state, "little")

    def setstate(self, state):
        # `mbidecoder_setstate` parses `(bytes, int)`; the pending buffer holds
        # at most MAXDECPENDING bytes.
        if not isinstance(state, tuple):
            raise TypeError(
                f"setstate() argument must be tuple, not {type(state).__name__}"
            )
        # `PyArg_ParseTuple(state, "Sn;setstate(): illegal state argument")`
        # names every shape failure with the one message its format string
        # carries, the wrong number of elements included.
        if (
            len(state) != 2
            or not isinstance(state[0], bytes)
            or not isinstance(state[1], int)
        ):
            raise TypeError("setstate(): illegal state argument")
        buffer, flag = state
        if len(buffer) > 8:
            raise UnicodeDecodeError(
                self.codec.name, buffer, 0, len(buffer), "pending buffer too large"
            )
        codec_state = int.to_bytes(flag, 8, "little")
        self.pending = buffer
        self.state = bytearray(codec_state)


class MultibyteIncrementalEncoder:
    errors = _errors_property
    _init_name = "IncrementalEncoder"
    _init_errors_position = 1

    def __init__(self, errors=_OMITTED):
        cls = type(self)
        self._errors = _errors_arg(cls._init_name, cls._init_errors_position, errors)
        self.codec = _codec_of(self)
        self.pending = ""
        # PyPy's `MultibyteIncrementalEncoder._initialize` owns one persistent
        # encodebuf.  Carry its `MultibyteCodec_State.c` explicitly so HZ's
        # ASCII/GB shift survives between calls.
        self.state = bytearray(_initial_state(self.codec.name, False))

    def encode(self, object, final=False):
        data = self.pending + object
        output, consumed = _encode_stateful(
            self.codec.name, data, self.errors, (final, self.state)
        )
        self.pending = data[consumed:]
        return output

    def reset(self):
        self.pending = ""
        self.state = bytearray(_initial_state(self.codec.name, False))

    def getstate(self):
        # PyPy `MultibyteIncrementalEncoder.getstate_w`: the state is one
        # little-endian integer over a 17-byte buffer:
        #   byte 0             length of the pending utf-8, 0..8
        #   bytes 1..1+length  the pending code points, utf-8
        #   the 8 bytes after  the codec state
        pending = self.pending.encode("utf-8")
        if len(pending) > 8:
            raise UnicodeError("pending buffer too large")
        buffer = bytes([len(pending)]) + pending + bytes(self.state)
        return int.from_bytes(buffer, "little")

    def setstate(self, state):
        # The `getstate` layout in reverse.  `TextIOWrapper.seek` restores the
        # initial state as the plain int 0.  A value that does not fit the
        # 17-byte buffer raises OverflowError out of `to_bytes`, the way
        # `_PyLong_AsByteArray` does in `mbiencoder_setstate`.
        if not isinstance(state, int):
            raise TypeError(
                f"setstate() argument must be int, not {type(state).__name__}"
            )
        buffer = int.to_bytes(state, 17, "little")
        pending_len = buffer[0]
        if pending_len > 8:
            raise UnicodeError("pending buffer too large")
        pending = buffer[1 : 1 + pending_len]
        try:
            decoded = pending.decode("utf-8")
        except UnicodeDecodeError as ex:
            # Neither half is stored when the pending buffer does not decode,
            # the way `mbiencoder_setstate` returns before both writes.
            raise UnicodeDecodeError(
                "utf-8",
                pending,
                ex.start,
                ex.start + 1,
                "invalid utf-8 in setstate pending buffer",
            ) from None
        self.pending = decoded
        self.state = bytearray(buffer[1 + pending_len : 9 + pending_len])


class MultibyteStreamReader(MultibyteIncrementalDecoder):
    _init_name = "StreamReader"
    _init_errors_position = 2

    def __new__(cls, stream, errors=_OMITTED):
        self = object.__new__(cls)
        self.stream = stream
        return self

    def __init__(self, stream, errors=_OMITTED):
        MultibyteIncrementalDecoder.__init__(self, errors)

    def __read(self, read, size):
        if size is None or size < 0:
            return MultibyteIncrementalDecoder.decode(self, read(), True)
        while True:
            data = read(size)
            final = not data
            output = MultibyteIncrementalDecoder.decode(self, data, final)
            if output or final:
                return output
            size = 1

    def read(self, size=None):
        return self.__read(self.stream.read, size)

    def readline(self, size=None):
        return self.__read(self.stream.readline, size)

    def readlines(self, sizehint=None):
        return self.__read(self.stream.read, sizehint).splitlines(True)


class MultibyteStreamWriter(MultibyteIncrementalEncoder):
    _init_name = "StreamWriter"
    _init_errors_position = 2

    def __new__(cls, stream, errors=_OMITTED):
        self = object.__new__(cls)
        self.stream = stream
        return self

    def __init__(self, stream, errors=_OMITTED):
        MultibyteIncrementalEncoder.__init__(self, errors)

    def write(self, data):
        self.stream.write(MultibyteIncrementalEncoder.encode(self, data))

    def reset(self):
        data = MultibyteIncrementalEncoder.encode(self, "", final=True)
        if data:
            self.stream.write(data)
        MultibyteIncrementalEncoder.reset(self)

    def writelines(self, lines):
        for data in lines:
            self.write(data)
