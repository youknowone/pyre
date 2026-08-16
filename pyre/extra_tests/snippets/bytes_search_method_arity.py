# pyre-check: gate=1
result = True
for constructor in (bytes, bytearray):
    value = constructor(b'hello')
    for name in ('find', 'rfind', 'index', 'rindex', 'count', 'startswith', 'endswith'):
        try:
            getattr(value, name)(constructor(b'x'), None, None, None)
        except TypeError as exc:
            result = result and name in str(exc)
        else:
            result = False

assert result
