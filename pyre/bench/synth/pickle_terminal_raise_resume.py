# pyre-check: spec-folds=load_attr,builtin_getattr
# pyre-check: max-wasm-ratio=6.3
# Function.call_args now enters every application-level callee through the
# recursive portal, matching PyPy's PyCode.funcrun -> PyFrame.run chain.  The
# oracle reaches 446 loops / 4 bridges in this pure-Python pickle workload;
# pyre's wasm run reaches 72 / 0, so suppressing those entry traces to recover
# the old timing would move away from the oracle.  wasm materializes the traces
# as separate modules, while this fixture also retains 450 host residual calls.
# The post-port ratios reproduce at 5.3x and 5.4x against dynasm on
# darwin-arm64; 6.3x is the highest observation plus WASM_RATIO_FIT_HEADROOM
# (15%).  One ubuntu-24.04 run read it at 3.5x, far enough under the 4x
# ceiling's fit margin that the summary there names the allowance
# outgrown -- it is the darwin reading no CI run measures that this holds
# for.
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
