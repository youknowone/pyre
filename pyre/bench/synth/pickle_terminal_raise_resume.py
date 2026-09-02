# pyre-check: spec-folds=load_attr,builtin_getattr
# pyre-check: max-wasm-ratio=5.2
# This threshold=1 fixture compiles 70 wasm loops against 30 native loops. On
# the final wasm-host module, four ratio probes reached 4.2x..4.5x; the runner's
# own census attributes 183.2ms to materializing 31 trace functions in wasmtime
# and only 40.7ms to all 436 guest/host residual crossings. Eliminating every
# crossing would therefore still not make the global 4x ceiling reliable here.
# The 5.2x allowance is the 4.5x high-water mark plus the required 15% fitting
# margin; the exact loop/abort/guard jitstats below remain independently gated.
# The attribute fold, 690 firings across the corpus and undeclared. Pure-Python
# pickle drives it 122 times here, more than any other fixture.  It also reaches
# `builtin_getattr`, which a corpus census found no other fixture firing: once
# here, out of 77 consultations.
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
