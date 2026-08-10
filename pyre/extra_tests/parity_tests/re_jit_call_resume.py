import re


for i in range(10_000):
    re.compile(str(i) + "|x")

print("OK")
# CPython-suite gap: re tests cannot cover a pyre JIT call abort and frame resume.
# parity-tests reason: this is a JIT-only operand-stack/frame-identity regression.
