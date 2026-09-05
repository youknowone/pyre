from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).with_name("check-rtyper-skip-subjects.py")
SPEC = importlib.util.spec_from_file_location("check_rtyper_skip_subjects", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class SkipSubjectRatchetTests(unittest.TestCase):
    @staticmethod
    def empty_skip_census() -> str:
        return (
            "=== majit decline census [analyze_pipeline]: 3 events, 1 rows ===\n"
            f"  {CHECKER.GATE}\n"
            "         3         3  match-real-rtyper (ACCEPT, not a decline)\n"
        )

    def test_gate_summary_makes_an_empty_skip_set_observable(self) -> None:
        census = self.empty_skip_census()
        self.assertEqual(CHECKER.parse(census), {})
        self.assertIsNotNone(CHECKER.GATE_SUMMARY.search(census))

    def test_an_absent_gate_is_not_mistaken_for_zero_skips(self) -> None:
        census = (
            "=== majit decline census [analyze_pipeline]: 0 events, 0 rows ===\n"
            "  (no instrumented gate recorded a decline in this process)\n"
        )
        self.assertIsNone(CHECKER.GATE_SUMMARY.search(census))

    def test_main_rejects_a_census_that_never_observed_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            stderr = root / "stderr"
            baseline = root / "baseline.txt"
            stderr.write_text(
                "=== majit decline census [analyze_pipeline]: 0 events, 0 rows ===\n"
                "  (no instrumented gate recorded a decline in this process)\n"
            )
            baseline.write_text("# corpus=abc platform=test\n")
            with (
                mock.patch.object(CHECKER, "scan_for_stderr", return_value=stderr),
                mock.patch.object(CHECKER, "baseline_path", return_value=baseline),
                mock.patch.object(CHECKER, "ROOT", root),
                mock.patch.object(sys, "argv", [str(SCRIPT), "--no-build"]),
            ):
                with self.assertRaisesRegex(SystemExit, "never observed the dual gate"):
                    CHECKER.main()

    def test_existing_empty_baseline_differs_from_missing_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "baseline.txt"
            exists, header, subjects = CHECKER.read_baseline(path)
            self.assertFalse(exists)
            self.assertEqual((header, subjects), ({}, {}))

            path.write_text("# corpus=abc platform=test\n")
            exists, header, subjects = CHECKER.read_baseline(path)
            self.assertTrue(exists)
            self.assertEqual(header, {"corpus": "abc", "platform": "test"})
            self.assertEqual(subjects, {})

    def test_main_accepts_an_existing_empty_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            stderr = root / "stderr"
            baseline = root / "baseline.txt"
            stderr.write_text(self.empty_skip_census())
            baseline.write_text("# corpus=abc platform=test\n")
            with (
                mock.patch.object(CHECKER, "scan_for_stderr", return_value=stderr),
                mock.patch.object(CHECKER, "baseline_path", return_value=baseline),
                mock.patch.object(CHECKER, "ROOT", root),
                mock.patch.object(CHECKER, "corpus_key", return_value=("abc", [])),
                mock.patch.object(CHECKER, "platform_key", return_value="test"),
                mock.patch.object(sys, "argv", [str(SCRIPT), "--no-build"]),
            ):
                self.assertEqual(CHECKER.main(), 0)

    def test_update_can_record_the_terminal_empty_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            stderr = root / "stderr"
            baseline = root / "baseline.txt"
            stderr.write_text(self.empty_skip_census())
            with (
                mock.patch.object(CHECKER, "scan_for_stderr", return_value=stderr),
                mock.patch.object(CHECKER, "baseline_path", return_value=baseline),
                mock.patch.object(CHECKER, "ROOT", root),
                mock.patch.object(CHECKER, "corpus_key", return_value=("abc", [])),
                mock.patch.object(CHECKER, "platform_key", return_value="test"),
                mock.patch.object(
                    sys, "argv", [str(SCRIPT), "--no-build", "--update"]
                ),
            ):
                self.assertEqual(CHECKER.main(), 0)
            exists, header, subjects = CHECKER.read_baseline(baseline)
            self.assertTrue(exists)
            self.assertEqual(header, {"corpus": "abc", "platform": "test"})
            self.assertEqual(subjects, {})


if __name__ == "__main__":
    unittest.main()
