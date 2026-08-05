# pyre-check: max-pypy-ratio=5
# pyre-check: min-pypy-ratio=0.55
# dir() lists lone-surrogate attribute/global names instead of dropping
# them, and never crashes reading a surrogate str key.  DictStorage keeps
# names as byte-ish str (entries_wtf8), so a name set via setattr with a
# lone-surrogate string survives the dir() name walk.

import sys

S1 = '\udc81'            # lone surrogate name
S2 = '\udc84'            # another lone surrogate name


class C:
    pass


# The walks are repeated so the measured body is larger than the process
# startup floor; only the last round's answers are printed, so the output is
# identical to a single round.
REPEAT = 100


def main():
    for _ in range(REPEAT):
        # Module global named by a lone surrogate appears in dir(module).
        setattr(sys, S1, 1)
        dm = dir(sys)
        mod_s1 = S1 in dm
        mod_argv = 'argv' in dm

        # Type attribute named by a surrogate appears in dir(type).
        setattr(C, S2, 3)
        dt = dir(C)
        type_s2 = S2 in dt

        # Instance attribute named by a surrogate appears in dir(instance),
        # alongside the surrogate name inherited from the type.
        c = C()
        setattr(c, S1, 4)
        dc = dir(c)
        inst_s1 = S1 in dc
        inst_s2 = S2 in dc

        # dir() output stays sorted with surrogate names present.
        type_sorted = dt == sorted(dt)

    print('mod_s1', mod_s1)
    print('mod_argv', mod_argv)
    print('type_s2', type_s2)
    print('inst_s1', inst_s1)
    print('inst_s2', inst_s2)
    print('type_sorted', type_sorted)


main()
