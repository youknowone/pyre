# pyre-check: platforms=win32
"""`winreg` is the registry, and the `*Ex` calls are the ones with the knobs.

`CreateKey`/`DeleteKey` take the key and the sub-key and nothing else, so a
caller that needs to say *which view* of the registry it means — the 64-bit one
or the 32-bit one a WOW64 process sees — has to reach for `CreateKeyEx` and
`DeleteKeyEx` and pass an access mask.  `setuptools`, `mimetypes` and
`webbrowser` all read the registry through those.

Everything here is written under `HKEY_CURRENT_USER\\Software`, which is the one
hive an ordinary account owns, and removed again.
"""

import sys
import winreg

# `winreg.error` is `OSError`, not a class of its own -- code that spells its
# handler `except winreg.error` catches exactly what the Reg* calls raise.
assert winreg.error is OSError

# The handle type is published under one name. It calls itself `PyHKEY`, which
# is what `repr` shows, but `winreg.PyHKEY` is not a name the module binds.
assert winreg.HKEYType.__name__ == "PyHKEY", winreg.HKEYType.__name__
assert winreg.HKEYType.__module__ == "winreg", winreg.HKEYType.__module__
assert not hasattr(winreg, "PyHKEY")

for name in ("CreateKeyEx", "DeleteKeyEx", "LoadKey", "SaveKey"):
    assert callable(getattr(winreg, name)), name

ROOT = winreg.HKEY_CURRENT_USER
BASE = r"Software\pyre-parity-%d" % id(winreg)
SUB = BASE + r"\child"


def cleanup():
    for path in (SUB, BASE):
        try:
            winreg.DeleteKey(ROOT, path)
        except OSError:
            pass


cleanup()
try:
    # CreateKeyEx makes the key and hands back a handle open under the mask it
    # was given -- KEY_WRITE here, so the value write below is allowed.
    key = winreg.CreateKeyEx(ROOT, BASE, 0, winreg.KEY_WRITE)
    assert isinstance(key, winreg.HKEYType), type(key)
    assert int(key) != 0
    assert bool(key) is True

    # The access mask is a keyword too, which is the spelling callers that skip
    # `reserved` use.
    nested = winreg.CreateKeyEx(ROOT, SUB, access=winreg.KEY_ALL_ACCESS)
    winreg.SetValueEx(nested, "answer", 0, winreg.REG_DWORD, 42)
    winreg.SetValueEx(nested, "name", 0, winreg.REG_SZ, "pyre")
    assert winreg.QueryValueEx(nested, "answer") == (42, winreg.REG_DWORD)
    assert winreg.QueryValueEx(nested, "name") == ("pyre", winreg.REG_SZ)
    nested.Close()
    assert bool(nested) is False

    # Creating a key that is already there opens it rather than failing.
    again = winreg.CreateKeyEx(ROOT, SUB, 0, winreg.KEY_READ)
    assert winreg.QueryValueEx(again, "answer") == (42, winreg.REG_DWORD)
    again.Close()

    # The parent now reports the child through QueryInfoKey/EnumKey.
    with winreg.OpenKeyEx(ROOT, BASE, 0, winreg.KEY_READ) as opened:
        sub_keys, values, _written = winreg.QueryInfoKey(opened)
        assert sub_keys == 1, sub_keys
        assert values == 0, values
        assert winreg.EnumKey(opened, 0) == "child"

    key.Close()

    # DeleteKeyEx takes the access mask where CreateKeyEx takes `reserved`, so
    # passing it positionally means the mask -- the two orders differ.
    winreg.DeleteKeyEx(ROOT, SUB, winreg.KEY_WOW64_64KEY, 0)
    try:
        winreg.OpenKeyEx(ROOT, SUB, 0, winreg.KEY_READ)
    except FileNotFoundError as exc:
        assert exc.winerror == 2, exc.winerror
    else:
        raise AssertionError("DeleteKeyEx left the key behind")

    # And by keyword, which is how a caller names the 32-bit view.
    winreg.DeleteKeyEx(ROOT, BASE, access=winreg.KEY_WOW64_64KEY)
    try:
        winreg.OpenKey(ROOT, BASE)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("DeleteKeyEx left the parent behind")

    # A key that is not there cannot be deleted, and says so with the code the
    # filesystem uses for the same thing.
    try:
        winreg.DeleteKeyEx(ROOT, BASE)
    except FileNotFoundError as exc:
        assert exc.winerror == 2, exc.winerror
    else:
        raise AssertionError("DeleteKeyEx removed a key that was not there")
finally:
    cleanup()

# SaveKey writes a hive file and needs SE_BACKUP_NAME to do it, which an
# ordinary process does not hold -- so either it refuses, or it succeeds and
# leaves a file. Both are answers; being absent or silently doing nothing is
# not.
import os
import shutil
import tempfile

hive = os.path.join(tempfile.mkdtemp(prefix="pyre_hive_"), "saved")
try:
    winreg.SaveKey(winreg.HKEY_CURRENT_USER, hive)
except OSError as exc:
    assert exc.winerror is not None, exc
else:
    assert os.path.exists(hive), hive
# A saved hive is not one file: the registry writes its transaction log beside
# it under names of its own choosing, so the directory is emptied rather than
# the one name removed.
shutil.rmtree(os.path.dirname(hive), ignore_errors=True)

# Both take strings, and neither takes a key that is not one.
for call, args in (
    (winreg.SaveKey, (ROOT, 0)),
    (winreg.LoadKey, (ROOT, "sub", 0)),
    (winreg.CreateKeyEx, (object(), "sub")),
    (winreg.DeleteKeyEx, (object(), "sub")),
):
    try:
        call(*args)
    except TypeError:
        pass
    else:
        raise AssertionError("%s accepted %r" % (call.__name__, args))

print("OK")
