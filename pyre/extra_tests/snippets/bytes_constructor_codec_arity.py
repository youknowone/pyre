# pyre-check: gate=1
result = True
for constructor in (bytes, bytearray):
    for args, argument, received in (
        (('', b'ascii'), 'encoding', 'bytes'),
        (('', 'ascii', b'ignore'), 'errors', 'bytes'),
        (('', None), 'encoding', 'None'),
        (('', 'ascii', None), 'errors', 'None'),
    ):
        try:
            constructor(*args)
        except TypeError as exc:
            expected = f"{constructor.__name__}() argument '{argument}' must be str, not {received}"
            result = result and str(exc) == expected
        else:
            result = False

assert result
