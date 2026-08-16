# pyre-check: gate=1
result = True
for value in (bytearray(), bytearray(b'preserved')):
    original = value.copy()
    try:
        value.resize((1 << 63) - 1)
    except MemoryError as exc:
        result = result and str(exc) == '' and value == original
    else:
        result = False

assert result
