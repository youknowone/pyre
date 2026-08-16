# pyre-check: gate=1
result = True
for value in (bytes(b'hello'), bytearray(b'hello')):
    for args in ((), (None, None), (None, b'', b'')):
        try:
            value.translate(*args)
        except TypeError:
            pass
        else:
            result = False
    try:
        value.translate(None, unexpected=b'')
    except TypeError:
        pass
    else:
        result = False

assert result
