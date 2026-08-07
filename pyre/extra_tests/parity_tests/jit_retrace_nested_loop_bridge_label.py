"""A retrace whose LABEL is assembled inside a bridge is branched to correctly.

A retrace is attached to a guard, so it is assembled through the bridge path and
carries its own LABEL. That LABEL publishes an in-buffer offset that only becomes
an absolute address when the target tokens are fixed up; a later trace closing
onto the same token reads the field and bakes it as a branch target. Skipping the
fixup on the bridge path made that branch jump to the raw offset.

The inner loop's accumulator flips int -> float partway through the outer loop,
which is what asks for the retrace.
"""

try:
    import pypyjit
except ImportError:
    pass
else:
    pypyjit.set_param("retrace_limit=5")


def accumulate(outer, inner):
    total = 0
    o = 0
    while o < outer:
        j = 0
        while j < inner:
            if o > 200:
                total = total + 0.5
            else:
                total = total + 1
            j += 1
        o += 1
    return total


assert accumulate(600, 100) == 40050.0

print("OK")
