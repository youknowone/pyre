import re


# 5_000, not more: the dynasm arm of this file spends about 27s of the runner's
# 30s per-fixture budget at 10_000, so it timed out on two different hosts.
# Halving it keeps what the file is for — 16 bridges either way, and 12 of the
# 15 single-frame blackhole adoptions the resume path is measured through.
for i in range(5_000):
    re.compile(str(i) + "|x")

print("OK")
# CPython-suite gap: re tests cannot cover a pyre JIT call abort and frame resume.
# parity-tests reason: this is a JIT-only operand-stack/frame-identity regression.
