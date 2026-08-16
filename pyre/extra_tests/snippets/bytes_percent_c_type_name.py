# pyre-check: gate=1
class Qualified:
    pass
Qualified.__module__ = 'package.module'

class Main:
    pass

messages = []
for format_string in (b'%c', '%c'):
    for value in (Qualified(), Main()):
        try:
            format_string % value
        except TypeError as exc:
            messages.append(str(exc))

result = (
    messages[0].endswith('not package.module.Qualified')
    and messages[1].endswith('not Main')
    and messages[2].endswith('not package.module.Qualified')
    and messages[3].endswith('not Main')
)

assert result
