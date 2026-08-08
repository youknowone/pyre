"""Which locale categories a platform has, and a setlocale that admits failure.

- The `LC_*` numbers belong to the host's C library, and POSIX and the MSVC CRT
  number them differently — `LC_ALL` is 6 on one and 0 on the other. A table of
  POSIX values published on Windows does not merely read wrong: every number
  names a *different* category there, so setting one sets another.
- `LC_MESSAGES` is a POSIX category the MSVC CRT has no counterpart for, and
  `locale.py` asks whether the name is here before using it.
- `setlocale` answered "C" for every name it was given wherever the real one
  was not being called, so a locale the host cannot install looked installed.
  `locale.Error` is how a caller finds out it is not — and code that asks for a
  locale in order to skip when it is missing gets the skip wrong otherwise.
"""

import locale
import sys


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def raises(call, exc):
    try:
        call()
    except exc as e:
        return e
    raise AssertionError(f"{exc.__name__} was not raised")


CATEGORIES = ["LC_ALL", "LC_COLLATE", "LC_CTYPE", "LC_MONETARY", "LC_NUMERIC",
              "LC_TIME"]

# ── which categories exist, and what they are numbered ────────────────────
for name in CATEGORIES:
    check(hasattr(locale, name), f"locale has no {name}")

values = [getattr(locale, name) for name in CATEGORIES]
check(len(set(values)) == len(values), f"two categories share a number: {values}")

# `LC_MESSAGES` is the one category whose presence is the platform's answer
# rather than a constant, so it is asserted as presence and not as a number.
check(
    hasattr(locale, "LC_MESSAGES") == (sys.platform != "win32"),
    f"LC_MESSAGES presence is wrong for {sys.platform}",
)

if sys.platform == "win32":
    # The MSVC CRT's own numbering, which is not POSIX's in any position.
    check(locale.LC_ALL == 0, locale.LC_ALL)
    check(locale.LC_COLLATE == 1, locale.LC_COLLATE)
    check(locale.LC_CTYPE == 2, locale.LC_CTYPE)
    check(locale.LC_MONETARY == 3, locale.LC_MONETARY)
    check(locale.LC_NUMERIC == 4, locale.LC_NUMERIC)
    check(locale.LC_TIME == 5, locale.LC_TIME)

# ── setlocale answers about the locale that was installed ─────────────────
# The one-argument form asks rather than sets, and what it answers has to be a
# name the two-argument form accepts back — the round trip is how `addCleanup`
# in a test, or a library restoring what it borrowed, puts the locale back.
for name in CATEGORIES:
    category = getattr(locale, name)
    current = locale.setlocale(category)
    check(isinstance(current, str) and current, f"setlocale({name}) -> {current!r}")
    check(locale.setlocale(category, current) == current, f"{name} did not round trip")

# "C" is the one locale every host has.
check(locale.setlocale(locale.LC_ALL, "C") == "C", "LC_ALL could not be set to C")

# A name no host can install is refused. Answering "C" here — the shape of a
# setlocale that never reached the C library — reports success for a locale
# that was not installed, and the caller then runs the work it meant to skip.
for absent in ("no-such-locale-xyz", "xx_YY.INVALID"):
    raises(lambda: locale.setlocale(locale.LC_CTYPE, absent), locale.Error)
    check(
        locale.setlocale(locale.LC_CTYPE) == "C",
        "a refused setlocale changed the installed locale",
    )

# An embedded NUL cannot reach a C string, and is a value error rather than a
# locale that could not be found.
raises(lambda: locale.setlocale(locale.LC_CTYPE, "C\0C"), ValueError)

print("OK")
