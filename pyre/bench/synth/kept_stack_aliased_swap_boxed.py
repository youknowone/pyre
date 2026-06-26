# Two operand-stack temps holding heap ints (>= 256) are swapped across a
# `goto_if_not` branch guard, a shape a per-jitcode merge-color map is prone
# to COLLAPSE (alias both slots onto one color).  The flat-free boxed-int
# kept-slot check (`kept_stack_has_boxed_int_hazard`) inspects each kept slot
# through its per-PC `pcdep_color_slots` color (or `const_ref_slots_at_pc`
# raw), so it must not be fooled by a collapsed/stale merge color the way the
# flat `stack_slot_color_map` read was.
#
# Regression guard for the kept_boxed_int flat-map migration: a wrong/stale
# color would either miss the boxed-int hazard (miscompile) or recover the
# wrong slot.  Pure arithmetic -> deterministic checksum.
N = 400000


def main():
    a = 100000
    b = 200000
    i = 0
    while i < N:
        if i % 6 == 0:
            a, b = 900000, a
        else:
            a, b = b, 800000
        a = a % 999983
        b = b % 999983
        i += 1
    print(a, b)


main()
