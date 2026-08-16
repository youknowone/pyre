# pyre-check: gate=1
import dis

source = (
    "i = 0\nacc = 0\nif i == 1:\n"
    + "    acc = acc + 1000\n" * 80
    + "while i < 6:\n    acc = acc + 1\n    i = i + 1\nr = acc\n"
)
code = compile(source, '<extended-arg>', 'exec')
assert any(i.opname == 'EXTENDED_ARG' for i in dis.get_instructions(code))

namespace = {}
exec(code, namespace)
assert namespace['r'] == 6
