# pyre-check: gate=1
# PyPy 5578: `_imp.extension_suffixes` advertises `.abi3.so` after the
# native suffix, and never a bare `.so`.  `site.py` installs the
# packaging.tags finder so pip will accept cp312+ abi3 wheels — but
# only when a loader exists.  CPython still lists a bare `.so`; this
# contract is PyPy/pyre only.
import sys
import textwrap
import types

if sys.implementation.name not in ("pypy", "pyre"):
    raise SystemExit(0)
if sys.platform == "win32":
    raise SystemExit(0)

import _imp
import importlib.machinery
import _pypy_abi3_tags

suffixes = _imp.extension_suffixes()
has_loader = ".abi3.so" in suffixes
finder_installed = any(
    isinstance(finder, _pypy_abi3_tags._Abi3TagsFinder) for finder in sys.meta_path
)

if has_loader:
    assert suffixes[0].endswith(".so"), suffixes
    assert ".so" not in suffixes, suffixes
    assert ".abi3.so" in importlib.machinery.EXTENSION_SUFFIXES
    assert finder_installed
else:
    # A loader-less build must not advertise abi3 to pip: that would
    # prefer an unloadable binary wheel over a working py3-none one.
    assert ".abi3.so" not in importlib.machinery.EXTENSION_SUFFIXES
    assert not finder_installed

FAKE_TAGS_PY = textwrap.dedent(
    """
    from collections import namedtuple
    Tag = namedtuple('Tag', 'interpreter abi platform')

    def platform_tags():
        return ['fakeplat']

    def compatible_tags(python_version=None, interpreter=None, platforms=None):
        platforms = list(platforms or platform_tags())
        for p in platforms:
            yield Tag('py3', 'none', p)
        if interpreter:
            yield Tag(interpreter, 'none', 'any')

    def sys_tags():
        yield Tag('pp312', 'pypy312_pp80', 'fakeplat')
        yield from compatible_tags(interpreter='pp3')
    """
)

fake = types.ModuleType("fake_packaging_tags")
exec(FAKE_TAGS_PY, fake.__dict__)
_pypy_abi3_tags.patch_tags_module(fake)
assert getattr(fake, _pypy_abi3_tags._PATCHED_ATTR)
patched = fake.compatible_tags
_pypy_abi3_tags.patch_tags_module(fake)
assert fake.compatible_tags is patched

major, minor = sys.version_info[:2]
version = "cp%d%d" % (major, minor)
result = list(
    fake.compatible_tags(
        interpreter="pp%d%d" % (major, minor), platforms=["p1", "p2"]
    )
)
abi3 = [t for t in result if t.abi == "abi3"]
expected_abi3 = [
    fake.Tag("cp%d%d" % (major, m), "abi3", platform)
    for m in range(minor, 11, -1)
    for platform in ("p1", "p2")
]
assert abi3 == expected_abi3, abi3
assert result[: len(abi3)] == abi3
assert result[len(abi3) :] == [
    fake.Tag("py3", "none", "p1"),
    fake.Tag("py3", "none", "p2"),
    fake.Tag("pp%d%d" % (major, minor), "none", "any"),
]

floor = list(fake.compatible_tags(interpreter="pp3", platforms=["p"]))
versions = {t.interpreter for t in floor if t.abi == "abi3"}
assert version in versions
assert "cp311" not in versions
assert "cp310" not in versions
assert all(int(v[3:]) >= 12 for v in versions)

for interpreter in ["pp311", "pp310", "cp312", "cp3"]:
    other = list(fake.compatible_tags(interpreter=interpreter, platforms=["p"]))
    assert not [t for t in other if t.abi == "abi3"], interpreter

none = list(fake.compatible_tags(interpreter=None, platforms=["p"]))
assert any(t.abi == "abi3" for t in none)

if sys.implementation.name == "pyre":
    for interpreter in [
        "pyre%d" % sys.version_info[0],
        "pyre%d%d" % sys.version_info[:2],
    ]:
        tagged = list(fake.compatible_tags(interpreter=interpreter, platforms=["p"]))
        found = [t for t in tagged if t.abi == "abi3"]
        assert found, interpreter
        assert all(t.interpreter.startswith("cp3") for t in found)

sys_tags = list(fake.sys_tags())
assert "%s-abi3-fakeplat" % version in ["-".join(t) for t in sys_tags]
assert sys_tags.index(fake.Tag("pp312", "pypy312_pp80", "fakeplat")) == 0
