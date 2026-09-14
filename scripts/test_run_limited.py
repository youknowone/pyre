"""Small resource-limit probes; run only inside a RAM-capped Linux VM."""
from pathlib import Path
import subprocess
import os
import re
import sys
import unittest
from unittest import mock
import importlib.util

RUNNER = Path(__file__).with_name("run-limited.py")


SUPPORTED_KERNEL = (sys.platform.startswith("linux") and
                    tuple(map(int, re.match(r"(\d+)\.(\d+)", os.uname().release).groups())) >= (4, 7))


class AdoptedChildren(unittest.TestCase):
    def test_missing_children_file_does_not_raise(self):
        spec = importlib.util.spec_from_file_location("limited_runner", RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with mock.patch.object(Path, "read_text", side_effect=FileNotFoundError):
            found = module.adopted_child_pids()
        self.assertEqual(found, [])


@unittest.skipUnless(SUPPORTED_KERNEL, "requires Linux >= 4.7 in a RAM-capped VM")
class Limits(unittest.TestCase):
    def run_guard(self, code, *limits):
        return subprocess.run(
            [sys.executable, str(RUNNER), "--seconds", "5", *limits, "--", sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10,
        )

    def test_immediately_detached_descendants_are_reaped(self):
        child = "import os,time; print('detachedpid=' + str(os.getpid()), flush=True); time.sleep(30)"
        code = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}], start_new_session=True)"
        # A pid file is written before detaching, so the parent need not wait
        # for our watchdog's first observation of the new process.
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "pid"
            code = f"import subprocess,sys,pathlib; p=subprocess.Popen([sys.executable,'-c',{child!r}], start_new_session=True); pathlib.Path({str(pidfile)!r}).write_text(str(p.pid))"
            result = self.run_guard(code)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(Path(f"/proc/{pidfile.read_text()}").exists())

    def load_runner(self):
        spec = importlib.util.spec_from_file_location("limited_runner", RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_reused_pid_is_not_in_historical_tree(self):
        runner = self.load_runner()
        rows = {10: (1, 10, 100, 1), 20: (1, 20, 99999, 3)}
        with mock.patch.object(runner.Path, "iterdir", return_value=[Path("10"), Path("20")]), mock.patch.object(runner, "process_info", side_effect=rows.get):
            found = runner.process_tree(10, {20: (10, 20, 10, 2)})
        self.assertEqual(set(found), {10})

    def test_census_failure_still_kills_and_waits(self):
        runner = self.load_runner()
        spawn = subprocess.Popen
        children = []
        def capture(*args, **kwargs):
            child = spawn(*args, **kwargs)
            children.append(child)
            return child
        argv = [str(RUNNER), "--seconds", "1", "--", sys.executable, "-c", "import time; time.sleep(30)"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(runner, "process_tree", side_effect=OSError("census failed")), mock.patch.object(subprocess, "Popen", side_effect=capture):
            with self.assertRaisesRegex(OSError, "census failed"):
                runner.main()
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].returncode)

    def test_tree_memory_limit(self):
        child = "import time; data=bytearray(32*1024*1024); time.sleep(30)"
        code = f"import subprocess,sys,time; children=[subprocess.Popen([sys.executable,'-c',{child!r}]) for _ in range(3)]; time.sleep(30)"
        result = self.run_guard(code, "--memory-mib", "96", "--process-mib", "64", "--seconds", "5")
        self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
        self.assertIn("RSS limit exceeded", result.stdout)

    def test_allocation_limit(self):
        result = self.run_guard("data=bytearray(128*1024*1024)", "--memory-mib", "96", "--process-mib", "64")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MemoryError", result.stdout)

    def test_virtual_reservation_does_not_relax_allocation_limit(self):
        result = self.run_guard("data=bytearray(128*1024*1024)",
                                "--memory-mib", "96", "--process-mib", "64",
                                "--address-space-mib", "8192")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MemoryError", result.stdout)

    def test_timeout(self):
        result = self.run_guard("while True: pass", "--seconds", "0.2")
        self.assertEqual(result.returncode, 124)
        self.assertIn("deadline exceeded", result.stdout)

    def test_output_limit(self):
        result = self.run_guard("import os\nwhile True: os.write(1,b'x'*65536)", "--output-mib", "1")
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(len(result.stdout), 34000)

    def test_memory_samples_are_retained(self):
        result = self.run_guard("import time; data=bytearray(8*1024*1024); time.sleep(1.2)")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        directory = Path(result.stdout.split("logs: ", 1)[1].split(";", 1)[0])
        lines = (directory / "rss.tsv").read_text().splitlines()
        self.assertEqual(lines[0], "seconds\tpid\tppid\trss_bytes")
        self.assertGreaterEqual(len(lines), 3)
        self.assertTrue(any(int(line.split("\t")[3]) >= 8*1024*1024 for line in lines[1:]))

    def test_build_artifact_budget_is_separate_from_output(self):
        result = self.run_guard(
            "import tempfile\nwith tempfile.TemporaryFile() as f: f.write(b'x'*(2*1024*1024))",
            "--output-mib", "1", "--file-mib", "4",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
