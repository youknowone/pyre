"""PyPy-compatible app-level surface over the cjkcodecs engine."""

# These are the module's own classes, so `__module__` should say so; without
# it the exec namespace leaves them looking like builtins.  Same reason
# `app_operator.py` sets it.
__name__ = "_multibytecodec"


# `PyArg_ParseTupleAndKeywords` leaves an argument the caller omitted at the C
# NULL its variable was initialized to, and `internal_error_callback(NULL)`
# reads that as "strict".  An explicit `None` reaches the `s` converter
# instead, which rejects it.  A sentinel is what keeps the two apart here.
_OMITTED = object()


def _codec_errors_arg(name, errors):
    # The `errors: str(accept={str, NoneType})` the two `MultibyteCodec`
    # methods declare: `None` is the "strict" spelling.
    if errors is None:
        return "strict"
    if not isinstance(errors, str):
        raise TypeError(
            f"{name}() argument 'errors' must be str or None, "
            f"not {type(errors).__name__}"
        )
    return errors


def _errors_arg(name, position, errors):
    # The plain `s` the four initializers declare: only a str, and an omitted
    # one means "strict".
    if errors is _OMITTED:
        return "strict"
    if not isinstance(errors, str):
        # `seterror` names `None` itself rather than its type.
        got = "None" if errors is None else type(errors).__name__
        raise TypeError(f"{name}() argument {position} must be str, not {got}")
    return errors


def _codec_of(obj):
    # `mbiencoder_init` and its three siblings read `codec` off the type, not
    # off the instance, so the four base classes -- which carry none -- raise
    # right here rather than at the first encode or decode.
    codec = type(obj).codec
    if not isinstance(codec, MultibyteCodec):
        raise TypeError("codec is unexpected type")
    return codec


class MultibyteCodec:
    def __init__(self, name):
        self.name = name

    def encode(self, input, errors=None):
        return _encode(self.name, input, _codec_errors_arg("encode", errors), True)

    def decode(self, input, errors=None):
        return _decode(self.name, input, _codec_errors_arg("decode", errors), True)


class MultibyteIncrementalDecoder:
    # The name and argument position the initializer reports a non-str
    # `errors` under; `MultibyteStreamReader` reuses the body under its own.
    _init_name = "IncrementalDecoder"
    _init_errors_position = 1

    def __init__(self, errors=_OMITTED):
        cls = type(self)
        self.errors = _errors_arg(cls._init_name, cls._init_errors_position, errors)
        self.codec = _codec_of(self)
        self.pending = b""
        self.state = 0

    def decode(self, object, final=False):
        data = self.pending + object
        output, consumed = _decode(self.codec.name, data, self.errors, final)
        self.pending = data[consumed:]
        return output

    def reset(self):
        self.pending = b""
        self.state = 0

    def getstate(self):
        return self.pending, self.state

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
        self.pending = buffer
        self.state = flag


class MultibyteIncrementalEncoder:
    _init_name = "IncrementalEncoder"
    _init_errors_position = 1

    def __init__(self, errors=_OMITTED):
        cls = type(self)
        self.errors = _errors_arg(cls._init_name, cls._init_errors_position, errors)
        self.codec = _codec_of(self)
        self.pending = ""
        # `MultibyteCodec_State`, the shift state a stateful codec threads
        # through its encode calls.  Every codec `_codecs_jp` carries is
        # stateless, so nothing here ever writes it -- `encreset` is NULL for
        # them, which is also why `reset` below leaves it alone -- but
        # `setstate` still has to give it back, so it is carried verbatim.
        self.state = bytes(8)

    def encode(self, object, final=False):
        data = self.pending + object
        output, consumed = _encode(self.codec.name, data, self.errors, final)
        self.pending = data[consumed:]
        return output

    def reset(self):
        self.pending = ""

    def getstate(self):
        # `interp_incremental.py:152-164`.  The state is one little-endian
        # integer over a 17-byte buffer:
        #   byte 0             length of the pending utf-8, 0..8
        #   bytes 1..1+length  the pending code points, utf-8
        #   the 8 bytes after  the codec state
        pending = self.pending.encode("utf-8")
        if len(pending) > 8:
            raise UnicodeError("pending buffer too large")
        buffer = bytes([len(pending)]) + pending + self.state
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
        buffer = state.to_bytes(17, "little")
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
        self.state = buffer[1 + pending_len : 9 + pending_len]


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
