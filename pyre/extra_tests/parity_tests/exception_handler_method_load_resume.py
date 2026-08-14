# CPython-suite gap: no suite test resumes a JIT frame mid-expression between a
# method load and the CALL that consumes it.
# parity-tests reason: an abort inside the inlined argument call flushes the
# caller frame back at that CALL, and the still-symbolic `out.append` method-load
# pair is part of the operand stack it has to write.  Publishing that slot as a
# null resumed the interpreter over a null callable (SIGSEGV in
# `classify_callable`); the slot has to arrive unresolved so the flush declines.


class Payload:
    def __init__(self, value):
        self.parts = value


def make_parts(value):
    return Payload(value).parts


def run(items):
    out = []
    for index in items:
        try:
            raise IndexError
        except IndexError:
            out.append(make_parts(index))
    return out


items = [index % 5 for index in range(60)]
for _ in range(400):
    assert run(items) == items

print("OK")
