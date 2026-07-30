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


# The stdlib implementation has the same shape, but its STOP handler ends in
# `raise _Stop(value)`.  Its terminal RAISE_VARARGS has no following Python
# instruction from which to synthesize a resume pivot: the JitCode's exact
# post-residual `live` marker must keep the already executed stack.pop() from
# being replayed after a trace abort.
import io
import pickle


payload = pickle.dumps(b"x", protocol=0)
for iteration in range(5_000):
    assert pickle._Unpickler(io.BytesIO(payload)).load() == b"x", iteration
