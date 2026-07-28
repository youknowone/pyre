# pyre-check: max-pypy-ratio=5
# dir() lists lone-surrogate attribute/global names instead of dropping
# them, and never crashes reading a surrogate str key.  DictStorage keeps
# names as byte-ish str (entries_wtf8), so a name set via setattr with a
# lone-surrogate string survives the dir() name walk.

import sys

S1 = '\udc81'            # lone surrogate name
S2 = '\udc84'            # another lone surrogate name


class C:
    pass


def main():
    # Module global named by a lone surrogate appears in dir(module).
    setattr(sys, S1, 1)
    dm = dir(sys)
    print('mod_s1', S1 in dm)
    print('mod_argv', 'argv' in dm)

    # Type attribute named by a surrogate appears in dir(type).
    setattr(C, S2, 3)
    dt = dir(C)
    print('type_s2', S2 in dt)

    # Instance attribute named by a surrogate appears in dir(instance),
    # alongside the surrogate name inherited from the type.
    c = C()
    setattr(c, S1, 4)
    dc = dir(c)
    print('inst_s1', S1 in dc)
    print('inst_s2', S2 in dc)

    # dir() output stays sorted with surrogate names present.
    print('type_sorted', dt == sorted(dt))


main()
