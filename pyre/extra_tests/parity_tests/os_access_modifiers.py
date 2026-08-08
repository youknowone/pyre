# pyre-check: platforms=linux,darwin
"""`os.access` honours dir_fd, effective_ids and follow_symlinks.

Named for these two platforms because `faccessat(2)` is what carries all three
modifiers, and Windows has none: the reference CPython refuses `dir_fd` there
with a NotImplementedError of its own and omits `access` from the same three
capability sets, so every assertion below would fail against a failing
reference.


All three are keyword-only modifiers that `access` has always *bound* — an
unknown keyword was an error, and a third positional was refused — while the
body called `access(2)`, which takes none of them.  A `dir_fd` was therefore
accepted and then ignored, so a relative name resolved against the working
directory instead of the descriptor, and `follow_symlinks=False` answered about
the file a symlink points at rather than the link.  Neither raised; both
answered confidently and wrongly.  `os.supports_dir_fd`,
`os.supports_effective_ids` and `os.supports_follow_symlinks` all omitted
`access`, which is the honest report of that, so a caller reading the
capability set before calling was told the truth and a caller passing the
modifier was not.

The modified call is one `faccessat(2)`; the plain call stays on `access(2)`.

The discriminating assertion is the pair below: the *same* relative name
answers True with `dir_fd` and False without it.  A test that only checked the
`dir_fd` form would have passed on the old build for any name that also
resolves from the working directory.

`effective_ids` cannot be discriminated by an unprivileged process — the real
and effective ids are equal, so both answers agree — so it is asserted to be
accepted and consistent rather than to change anything.  What pins it is
membership in `os.supports_effective_ids`, which is the same `HAVE_FACCESSAT`
bit the other two read.
"""

import os
import sys
import tempfile

ERRORS = []


def check(cond, what):
    if not cond:
        ERRORS.append(what)


def raises(what, exc, expected, fn):
    try:
        fn()
    except exc as e:
        check(str(e) == expected, f"{what}: got {str(e)!r}, expected {expected!r}")
        return
    except BaseException as e:
        ERRORS.append(f"{what}: raised {type(e).__name__}({e}), expected {exc.__name__}")
        return
    ERRORS.append(f"{what}: no exception, expected {exc.__name__}")


def check_class(what, exc, fn):
    """Assert fn() raises exactly `exc`, without pinning the message."""
    try:
        fn()
    except BaseException as e:
        if type(e) is not exc:
            ERRORS.append(
                f"{what}: raised {type(e).__name__}({e}), expected {exc.__name__}"
            )
        return
    ERRORS.append(f"{what}: no exception, expected {exc.__name__}")


# ── the capability sets say which modifiers work ──────────────────────────
check(os.access in os.supports_dir_fd, "os.access missing from supports_dir_fd")
check(
    os.access in os.supports_effective_ids,
    "os.access missing from supports_effective_ids",
)
check(
    os.access in os.supports_follow_symlinks,
    "os.access missing from supports_follow_symlinks",
)

root = tempfile.mkdtemp()
plain = os.path.join(root, "plain")
with open(plain, "w"):
    pass
sub = os.path.join(root, "sub")
os.mkdir(sub)
nested = os.path.join(sub, "g")
with open(nested, "w"):
    pass
dangling = os.path.join(root, "dangling")
os.symlink(os.path.join(root, "no-such-target"), dangling)

# The relative names below must not also resolve from the working directory,
# or the dir_fd assertions would pass without dir_fd doing anything.
check(not os.path.exists("sub/g"), "the test's relative name exists in the cwd")
check(not os.path.exists("dangling"), "the test's link name exists in the cwd")

dir_fd = os.open(root, os.O_RDONLY)
try:
    # ── dir_fd actually resolves the name ─────────────────────────────────
    check(
        os.access("sub/g", os.F_OK, dir_fd=dir_fd) is True,
        "access(dir_fd=) did not find a name under the descriptor",
    )
    check(
        os.access("sub/g", os.F_OK) is False,
        "the relative name resolved without dir_fd, so the pair proves nothing",
    )
    check(
        os.access("sub/no-such-file", os.F_OK, dir_fd=dir_fd) is False,
        "access(dir_fd=) claimed a missing name exists",
    )
    # An absolute name ignores the descriptor, so even a closed one is fine.
    check(
        os.access(plain, os.F_OK, dir_fd=dir_fd) is True,
        "an absolute name stopped resolving when dir_fd was given",
    )
    # `dir_fd=None` is the default spelled out, not a descriptor.
    check(
        os.access(plain, os.F_OK, dir_fd=None) is True,
        "dir_fd=None was not treated as absent",
    )

    # ── follow_symlinks asks about the link itself ────────────────────────
    check(
        os.access(dangling, os.F_OK) is False,
        "a dangling link reported its own existence through the target",
    )
    check(
        os.access(dangling, os.F_OK, follow_symlinks=False) is True,
        "access(follow_symlinks=False) did not see the link itself",
    )
    # Both modifiers at once, so the flag word carries two bits.
    check(
        os.access("dangling", os.F_OK, dir_fd=dir_fd, follow_symlinks=False) is True,
        "dir_fd and follow_symlinks=False together did not find the link",
    )
    check(
        os.access("dangling", os.F_OK, dir_fd=dir_fd) is False,
        "dir_fd with the default follow_symlinks resolved a dangling link",
    )

    # ── effective_ids is accepted and consistent ──────────────────────────
    check(
        os.access(plain, os.R_OK, effective_ids=True)
        == os.access(plain, os.R_OK),
        "effective_ids changed the answer for a process whose ids are equal",
    )
    check(
        os.access("sub/g", os.F_OK, dir_fd=dir_fd, effective_ids=True) is True,
        "effective_ids with dir_fd lost the descriptor",
    )

    # ── the mode is still the mode ────────────────────────────────────────
    # A modifier that reached the syscall as a mode bit, or a mode that got
    # dropped on the way, would show up here.
    check(os.access(plain, os.R_OK) is True, "a readable file reported unreadable")
    check(
        os.access(plain, os.X_OK) is False,
        "a file with no execute bit reported executable",
    )
    os.chmod(plain, 0o755)
    check(
        os.access(plain, os.X_OK) is True,
        "the execute bit did not become visible after chmod",
    )
    check(
        os.access("sub/g", os.R_OK | os.W_OK, dir_fd=dir_fd) is True,
        "a combined mode through dir_fd answered False",
    )

    # ── a bytes path still works ──────────────────────────────────────────
    check(
        os.access(os.fsencode(plain), os.F_OK) is True,
        "a bytes path stopped resolving",
    )

    # ── the modifiers are still type-checked ──────────────────────────────
    raises(
        "dir_fd given a str",
        TypeError,
        "argument should be integer or None, not str",
        lambda: os.access("sub/g", os.F_OK, dir_fd="x"),
    )
    raises(
        "a descriptor passed as the path",
        TypeError,
        "access: path should be string, bytes or os.PathLike, not int",
        lambda: os.access(dir_fd, os.F_OK),
    )

    # ── the parameters convert in declaration order ───────────────────────
    # Every parameter can raise, so which one reports first is observable.
    # `mode` is converted before any modifier is touched: giving both a bad
    # mode and a bad modifier must report the mode.  Only the exception
    # *class* is asserted, because the two interpreters word the integer
    # conversion differently for reasons that have nothing to do with access.
    class RaisingBool:
        def __bool__(self):
            raise RuntimeError("__bool__ ran")

    check_class(
        "a bad mode reports before a bad dir_fd",
        OverflowError,
        lambda: os.access("sub/g", 2**40, dir_fd="not a descriptor"),
    )
    check_class(
        "a bad mode reports before a flag's __bool__ runs",
        OverflowError,
        lambda: os.access("sub/g", 2**40, effective_ids=RaisingBool()),
    )
    # ...and with an acceptable mode the flag *is* consulted, so the two
    # assertions above are not passing because the flags stopped being read.
    check_class(
        "a flag's __bool__ runs once the mode converts",
        RuntimeError,
        lambda: os.access("sub/g", os.F_OK, effective_ids=RaisingBool()),
    )
finally:
    os.close(dir_fd)
    os.unlink(dangling)
    os.unlink(nested)
    os.rmdir(sub)
    os.unlink(plain)
    os.rmdir(root)

if ERRORS:
    for e in ERRORS:
        print("FAIL:", e, file=sys.stderr)
    raise AssertionError(f"{len(ERRORS)} divergence(s)")

print("OK")
