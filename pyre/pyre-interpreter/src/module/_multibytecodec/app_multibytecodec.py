"""PyPy-compatible app-level surface over the cjkcodecs engine."""


class MultibyteCodec:
    def __init__(self, name):
        self.name = name

    def encode(self, input, errors="strict"):
        return _encode(self.name, input, errors, True)

    def decode(self, input, errors="strict"):
        return _decode(self.name, input, errors, True)


class MultibyteIncrementalDecoder:
    def __init__(self, errors="strict"):
        self.errors = "strict" if errors is None else errors
        self.pending = b""

    def decode(self, object, final=False):
        data = self.pending + object
        output, consumed = _decode(self.codec.name, data, self.errors, final)
        self.pending = data[consumed:]
        return output

    def reset(self):
        self.pending = b""

    def getstate(self):
        return self.pending, 0

    def setstate(self, state):
        self.pending = state[0]


class MultibyteIncrementalEncoder:
    def __init__(self, errors="strict"):
        self.errors = "strict" if errors is None else errors
        self.pending = ""

    def encode(self, object, final=False):
        data = self.pending + object
        output, consumed = _encode(self.codec.name, data, self.errors, final)
        self.pending = data[consumed:]
        return output

    def reset(self):
        self.pending = ""

    def getstate(self):
        return self.pending or 0

    def setstate(self, state):
        # TextIOWrapper uses the initial integer state after seeking away
        # from a pending JIS X 0213 composition.
        self.pending = "" if state == 0 else state


class MultibyteStreamReader(MultibyteIncrementalDecoder):
    def __new__(cls, stream, errors=None):
        self = object.__new__(cls)
        self.stream = stream
        return self

    def __init__(self, stream, errors=None):
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
    def __new__(cls, stream, errors=None):
        self = object.__new__(cls)
        self.stream = stream
        return self

    def __init__(self, stream, errors=None):
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
