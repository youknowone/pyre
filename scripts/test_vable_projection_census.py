from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "pyre"
    / "scripts"
    / "vable-projection-census.py"
)
SPEC = importlib.util.spec_from_file_location("vable_projection_census", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CENSUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CENSUS)

FIELDS = ["valuestackdepth", "pycode", "last_instr", "debugdata", "locals_cells_stack_w"]


class VableProjectionCensusTests(unittest.TestCase):
    def test_deref_of_mut_self_is_not_counted(self) -> None:
        dump = (
            "pub fn settopvalue<'_0>(self: &'_0 mut PyFrame, "
            "value: *mut PyObject, index_from_top: usize)\n"
            "{\n"
            "        _5 = copy (*self).valuestackdepth;\n"
            "        _14 = copy (*self).valuestackdepth;\n"
            "}\n"
        )
        counts, unclassified = CENSUS.census(dump, FIELDS)
        self.assertEqual(counts, {})
        self.assertEqual(unclassified, [])

    def test_panic_message_is_not_a_projection(self) -> None:
        dump = (
            "pub fn settopvalue<'_0>(self: &'_0 mut PyFrame, "
            "value: *mut PyObject, index_from_top: usize)\n"
            "{\n"
            "        _5 = copy (*self).valuestackdepth;\n"
            "        _15 = panic(const "
            '"assertion failed: index < self.valuestackdepth") '
            "-> bb8 (unwind: bb9);\n"
            "}\n"
        )
        counts, unclassified = CENSUS.census(dump, FIELDS)
        self.assertEqual(counts, {})
        self.assertEqual(unclassified, [])

    def test_by_value_frame_constructor_is_counted(self) -> None:
        dump = (
            "pub fn new(frame: PyFrame)\n"
            "{\n"
            "        _1 = copy (frame_1).pycode;\n"
            "        frame_1.valuestackdepth = const 0usize;\n"
            "}\n"
        )
        counts, unclassified = CENSUS.census(dump, FIELDS)
        self.assertEqual(dict(counts), {"new": 2})
        self.assertEqual(unclassified, [])

    def test_comment_is_not_a_projection(self) -> None:
        dump = (
            "pub fn new(frame: PyFrame)\n"
            "{\n"
            "        //   self.valuestackdepth = code.co_nlocals\n"
            "}\n"
        )
        counts, unclassified = CENSUS.census(dump, FIELDS)
        self.assertEqual(counts, {})
        self.assertEqual(unclassified, [])


if __name__ == "__main__":
    unittest.main()
