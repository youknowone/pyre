# pyre-check: gate=1
names = (
    'capitalize', 'lower', 'upper', 'swapcase', 'title',
    'isdigit', 'isalpha', 'isalnum', 'isspace', 'isascii',
    'isupper', 'islower', 'istitle',
)
result = True
for value in (b'x', bytearray(b'x')):
    owner = type(value).__name__
    for name in names:
        try:
            getattr(value, name)(42)
        except TypeError as exc:
            result = result and str(exc) == (
                owner + '.' + name + '() takes no arguments (1 given)'
            )
        else:
            result = False

assert result
