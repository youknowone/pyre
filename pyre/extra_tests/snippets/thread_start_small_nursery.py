# A moving collection during `_thread` bootstrap used to leave the worker
# calling a nursery word that had been reused. `Bootstrapper.bootstrap`
# copies `w_callable` only after `gc_thread_start`, onto that thread's
# shadow stack.
import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import threading

    class Box:
        def __init__(self, i):
            self.i = i
            self.junk = [str(j) for j in range(30)]

        def run(self, out):
            for _k in range(200):
                x = [Box.__new__(Box) for _ in range(20)]
            out.append(self.i)

    def main():
        out = []
        for _rnd in range(20):
            ts = [threading.Thread(target=Box(i).run, args=(out,)) for i in range(8)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        print(len(out), flush=True)

    main()
    """
)

env = os.environ.copy()
env["PYRE_JIT"] = "0"
env["PYPY_GC_NURSERY"] = "64k"
proc = subprocess.run(
    [sys.executable, "-c", SCRIPT],
    env=env,
    capture_output=True,
    text=True,
    timeout=180,
)
assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
assert proc.stdout.strip() == "160"
