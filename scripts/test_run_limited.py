"""Small resource-limit probes; run only inside a RAM-capped Linux VM."""
from pathlib import Path
import subprocess
import sys
import unittest

RUNNER = Path(__file__).with_name("run-limited.py")


@unittest.skipUnless(sys.platform.startswith("linux"), "use a RAM-capped Linux VM")
class Limits(unittest.TestCase):
    def run_guard(self, code, *limits):
        return subprocess.run(
            [sys.executable, str(RUNNER), *limits, "--", sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10,
        )

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
