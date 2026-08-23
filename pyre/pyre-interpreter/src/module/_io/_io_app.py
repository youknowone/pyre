"""App-level fallbacks for the _io module.

IncrementalNewlineDecoder remains implemented here.

`_TextIOBase` is bound by the module initializer before this source runs.
"""


class IncrementalNewlineDecoder:
    r"""Codec used when reading a file in universal newlines mode.  It wraps
    another incremental decoder, translating \r\n and \r into \n.  It also
    records the types of newlines encountered.  When used with
    translate=False, it ensures that the newline sequence is returned in
    one piece.

    `_io.IncrementalNewlineDecoder` is a standalone type, not a
    `codecs.IncrementalDecoder` subclass; `decode_source` and `TextIOWrapper`
    construct it with `decoder=None` to translate an already-decoded string.
    """

    _LF = 1
    _CR = 2
    _CRLF = 4

    def __init__(self, decoder, translate, errors="strict"):
        if errors is None:
            errors = "strict"
        elif not isinstance(errors, str):
            raise TypeError(
                "TextIOWrapper() argument 'errors' must be str or None, not %s"
                % type(errors).__name__
            )
        else:
            # io_check_errors minus the dev-mode handler lookup — a codecs
            # import here would recurse through decode_source.
            errors.encode("utf-8", "strict")
        if not isinstance(translate, int):
            try:
                translate = translate.__index__()
            except AttributeError:
                raise TypeError(
                    "'%s' object cannot be interpreted as an integer"
                    % type(translate).__name__
                ) from None
        self.errors = errors
        self.translate = translate
        self.decoder = decoder
        self.seennl = 0
        self.pendingcr = False

    def decode(self, input, final=False):
        # decode input (with the eventual \r from a previous pass)
        if self.decoder is None:
            output = input
        else:
            output = self.decoder.decode(input, final)
        if not isinstance(output, str):
            raise TypeError("decoder should return a string result")
        if self.pendingcr and (output or final):
            output = "\r" + output
            self.pendingcr = False

        # retain last \r even when not translating data:
        # then readline() is sure to get \r\n in one pass
        if output.endswith("\r") and not final:
            output = output[:-1]
            self.pendingcr = True

        # Record which newlines are read
        crlf = output.count("\r\n")
        cr = output.count("\r") - crlf
        lf = output.count("\n") - crlf
        self.seennl |= (
            (lf and self._LF) | (cr and self._CR) | (crlf and self._CRLF)
        )

        if self.translate:
            if crlf:
                output = output.replace("\r\n", "\n")
            if cr:
                output = output.replace("\r", "\n")

        return output

    def getstate(self):
        if self.decoder is None:
            buf = b""
            flag = 0
        else:
            buf, flag = self.decoder.getstate()
        flag <<= 1
        if self.pendingcr:
            flag |= 1
        return buf, flag

    def setstate(self, state):
        buf, flag = state
        self.pendingcr = bool(flag & 1)
        if self.decoder is not None:
            self.decoder.setstate((buf, flag >> 1))

    def reset(self):
        self.seennl = 0
        self.pendingcr = False
        if self.decoder is not None:
            self.decoder.reset()

    @property
    def newlines(self):
        return (
            None,
            "\n",
            "\r",
            ("\r", "\n"),
            "\r\n",
            ("\n", "\r\n"),
            ("\r", "\r\n"),
            ("\r", "\n", "\r\n"),
        )[self.seennl]
