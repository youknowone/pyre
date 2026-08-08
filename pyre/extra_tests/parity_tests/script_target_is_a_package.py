"""A directory or a zipfile named as the script argument runs its `__main__`.

`app_main.py:1054-1105` walks `sys.path_hooks` with the script filename; when a
hook accepts it the path goes on `sys.path[0]` and runpy runs the `__main__`
module inside it, instead of the file being read as source. The `sys.path[0]`
entry is prepended for a package target even under the safe-path flags, which
suppress it for an ordinary script.
"""

import os
import subprocess
import sys
import tempfile
import zipfile

PROBE = """\
import sys
print(sys.argv[0])
print(sys.path[0])
print(__name__)
print(__file__)
"""


def run(*args):
    child = subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert child.returncode == 0, (args, child.returncode, child.stderr)
    return child.stdout.splitlines()


with tempfile.TemporaryDirectory() as tmp:
    package_dir = os.path.join(tmp, "pkgdir")
    os.mkdir(package_dir)
    with open(os.path.join(package_dir, "__main__.py"), "w", encoding="utf-8") as f:
        f.write(PROBE)

    archive = os.path.join(tmp, "app.zip")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("__main__.py", PROBE)

    for target in (package_dir, archive):
        argv0, path0, name, file = run(target)
        assert argv0 == target, (target, argv0)
        assert path0 == os.path.abspath(target), (target, path0)
        assert name == "__main__", (target, name)
        assert file == os.path.join(target, "__main__.py"), (target, file)

        # `-P` and `-I` drop the script's own directory from `sys.path`, but a
        # package target is still prepended — it is what runpy resolves
        # `__main__` against, so dropping it would leave nothing to run.
        for flag in ("-P", "-I"):
            argv0, path0, name, _ = run(flag, target)
            assert argv0 == target, (flag, target, argv0)
            assert path0 == os.path.abspath(target), (flag, target, path0)
            assert name == "__main__", (flag, target, name)

    # A plain source file still takes the file path: `sys.path[0]` is the
    # script's directory, not the script.
    script = os.path.join(tmp, "plain.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(PROBE)
    argv0, path0, name, file = run(script)
    assert argv0 == script, argv0
    # Compared through `realpath`: the script directory entry is resolved on a
    # platform whose temporary directory is itself a symlink.
    assert os.path.realpath(path0) == os.path.realpath(os.path.dirname(script)), path0
    assert name == "__main__", name
    assert file == os.path.abspath(script), file

    # A directory with no `__main__.py` is still accepted by the path hook, so
    # the failure is runpy's missing-module error, not a read error.
    empty = os.path.join(tmp, "empty")
    os.mkdir(empty)
    child = subprocess.run(
        [sys.executable, empty],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert child.returncode == 1, child.returncode
    assert "__main__" in child.stderr, child.stderr

print("OK")
