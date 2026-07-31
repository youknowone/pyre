# Pure-Python pickle terminates its unpickler loop by raising the private
# `_Stop` exception from `load_stop`.  The JIT's after-residual guard snapshot
# must resume from the emitted post-call `-live-` anchor even though the
# semantic Python fallthrough is past the end of `load_stop`; replaying from
# the call pops the result a second time.
try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass

import pickle


checksum = 0
for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    # Keep every protocol and terminal `_Stop` path hot without turning this
    # resume regression into a separate polymorphic-trace stress test.
    for value in range(8):
        original = bytes([value & 0xFF]) * (1 + value // 256)
        payload = pickle._dumps(original, protocol)
        restored = pickle._loads(payload)
        if restored != original:
            raise AssertionError((protocol, value, restored, original))
        checksum += len(restored) + restored[0]

print("checksum =", checksum)
