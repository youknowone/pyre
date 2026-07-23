"""An inlined raising callee must retain its own concrete frame identity."""


class StopDispatch(Exception):
    pass


class Dispatcher:
    def __init__(self):
        self.position = 0
        self.stack = []

    def push(self):
        self.stack.append(42)

    def stop(self):
        raise StopDispatch(self.stack.pop())

    dispatch = {0: push, 1: stop}

    def load(self):
        try:
            while True:
                if self.position >= 2:
                    raise EOFError((self.position, self.stack))
                key = self.position
                self.position += 1
                self.dispatch[key](self)
        except StopDispatch as exc:
            return exc.args[0]


# This is the pure-Python pickle Unpickler dispatch shape: a hot caller loop
# invokes unbound methods from a table, and the STOP handler mutates a stack
# immediately before raising the exception caught by the caller.
for iteration in range(10_000):
    assert Dispatcher().load() == 42, iteration
