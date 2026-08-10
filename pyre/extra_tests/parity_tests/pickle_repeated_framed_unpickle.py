# CPython-suite gap: pickle tests omit this repeated framed-load hot/GC shape.
# parity-tests reason: this guards pyre's JIT carrier state and moving-GC roots.

"""Trace cancellation must preserve every live Python frame.

RPython's ``pyjitpl.py:2949-2955`` converts the complete MIFrame stack into
blackhole interpreters when tracing is cancelled.  The caller therefore stays
paused at its CALL while the callee continues from its own frame; it cannot be
collapsed back onto the caller's loop header.
"""

import io
import pickle


# A framed protocol-4 GLOBAL payload exercises _Unpickler.load's dispatch
# loop, load_short_binunicode's exception-covered read calls, and
# _Unframer.read.  Warming that path used to abort tracing after consuming the
# string-length byte, then collapse the live callee back onto the load loop's
# header.  Replaying from there interpreted the length (8) as an opcode.
payload = (
    b"\x80\x04\x95\x1b\x00\x00\x00\x00\x00\x00\x00"
    b"\x8c\x08builtins\x94\x8c\x0aissubclass\x94\x93\x94."
)

for _ in range(2000):
    restored = pickle._Unpickler(io.BytesIO(payload)).load()
    assert restored is issubclass

print("OK")
