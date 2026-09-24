# pyre-check: gate=1
# A moving collection between publishing a thread target and the child
# pinning it used to leave the child holding a reused nursery word.
# `os_thread.py` `Bootstrapper` keeps that data in a traced slot instead.
import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import threading
    from enum import Flag, auto

    class TestFlag(Flag):
        one = auto()
        two = auto()
        three = auto()
        four = auto()
        five = auto()
        six = auto()
        seven = auto()
        eight = auto()

    def cycle():
        for i in range(256):
            TestFlag(i)

    threads = [threading.Thread(target=cycle) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("ok", flush=True)
    """
)

env = os.environ.copy()
env["PYRE_NO_JIT"] = "1"
env["PYPY_GC_NURSERY"] = "1k"
env["MAJIT_STRICT"] = "1"
proc = subprocess.run(
    [sys.executable, "-c", SCRIPT],
    env=env,
    capture_output=True,
    text=True,
    timeout=120,
)
assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
assert "ok" in proc.stdout
