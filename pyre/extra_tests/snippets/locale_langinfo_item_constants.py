# pyre-check: gate=1
# pyre-check: platforms=darwin,linux
# CPython-suite gap: test__locale guards every `nl_langinfo` assertion behind
# `hasattr(locale, name)`, so a module that publishes none of the item keys
# skips the whole file.
# parity-tests reason: `_locale.nl_langinfo` accepted any key, but 54 of the
# 55 names `_localemodule.c langinfo_constants` publishes were absent, so the
# only key that could be spelled was `CODESET`.

# Windows has no `<langinfo.h>` and no `nl_langinfo`, so the whole family is
# absent there on every interpreter; the test is scoped to the hosts that
# carry it.

# `<langinfo.h>` numbers the items per platform — the BSD headers count from
# zero while glibc packs the category into the high bits — so the test reads
# the numbers back from the module rather than pinning them, and checks the
# key each name stands for by what `nl_langinfo` answers with.

import locale

NAMES = ["CODESET", "D_T_FMT", "D_FMT", "T_FMT", "T_FMT_AMPM", "AM_STR", "PM_STR"]
NAMES += [f"DAY_{i}" for i in range(1, 8)]
NAMES += [f"ABDAY_{i}" for i in range(1, 8)]
NAMES += [f"MON_{i}" for i in range(1, 13)]
NAMES += [f"ABMON_{i}" for i in range(1, 13)]
NAMES += ["ERA", "ERA_D_FMT", "ERA_D_T_FMT", "ERA_T_FMT", "ALT_DIGITS"]
NAMES += ["RADIXCHAR", "THOUSEP", "YESEXPR", "NOEXPR", "CRNCYSTR"]

missing = [name for name in NAMES if not hasattr(locale, name)]
assert not missing, missing

keys = {name: getattr(locale, name) for name in NAMES}
assert len(set(keys.values())) == len(NAMES), keys
for name, key in keys.items():
    assert isinstance(key, int), (name, key)
    assert isinstance(locale.nl_langinfo(key), str), name

# `YESSTR` and `NOSTR` are the two `<langinfo.h>` items the module leaves out.
assert not hasattr(locale, "YESSTR")
assert not hasattr(locale, "NOSTR")

# The C locale is what the interpreter starts in, so these are its answers.
assert locale.setlocale(locale.LC_TIME) == "C"
assert locale.nl_langinfo(locale.DAY_1) == "Sunday"
assert locale.nl_langinfo(locale.ABDAY_1) == "Sun"
assert locale.nl_langinfo(locale.MON_1) == "January"
assert locale.nl_langinfo(locale.ABMON_12) == "Dec"
assert locale.nl_langinfo(locale.RADIXCHAR) == "."
assert locale.nl_langinfo(locale.YESEXPR) == "^[yY]"
assert locale.nl_langinfo(locale.NOEXPR) == "^[nN]"
assert locale.nl_langinfo(locale.ERA) == ""

print("OK")
