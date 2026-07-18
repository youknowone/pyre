# A single hot function whose int register + constant-pool slot count exceeds the
# 256-entry ceiling of the single-byte JitCode operand encoding (assembler.py chr()).
# Such a trace cannot be encoded as single-byte slot operands, so it must be
# DECLINED (trace compilation aborted, the interpreter keeps running) rather than
# asserting mid-encode and panicking the process. The trace-jitcode builder now
# propagates the over-cap condition up as a "cannot compile" result and the caller
# drops the trace, exactly as it does for an over-long trace. Deterministic,
# terminating, prints an int checksum; jit == nojit because the trace declines.
M = 1000000007

# Generate a body with >256 distinct integer constants so the trace's int
# register+const-pool count crosses the 256 single-byte slot ceiling.
_lines = ["def f(i):", "    a = (i * 3 + 1) % M", "    q = a"]
for k in range(300):
    _lines.append("    q = (a + {}) % M".format(100000 + k * 11))
_lines.append("    return (q + a) % M")
exec("\n".join(_lines))

acc = 0
for i in range(8000):
    acc = (acc + f(i)) % M
print(acc)
