from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("check-gc-root-brackets.py")
SPEC = importlib.util.spec_from_file_location("check_gc_root_brackets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

# Shape CI printed for `pyre-module.ullbc`: opaque `pyre_object` means no
# PyObjectRef type id, so the liveness columns are omitted.
SKIPPED_LIVENESS = """\
== pyre_module (282 fun_decls) ==
   collecting-alloc seeds: 10 seed function(s); UNMATCHED patterns: ["majit_gc::standalone_alloc_nursery_collecting_typed_rooted", "majit_gc::standalone_alloc_fast_nursery_collecting_typed_rooted"]
        cannot reach any collection        : 0
    (no PyObjectRef type id found — liveness scan skipped)
"""


class GcRootParseTests(unittest.TestCase):
    def test_skipped_liveness_is_zero_not_a_shape_error(self) -> None:
        got = CHECKER.parse(SKIPPED_LIVENESS)
        self.assertEqual(got["brackets_reaching_no_collection"], 0)
        self.assertEqual(got["unbracketed_calls"], 0)
        self.assertEqual(got["tier1_calls"], 0)
        self.assertEqual(got["tier15_calls"], 0)
        self.assertEqual(got["frames_across_collecting"], 0)
        self.assertEqual(got["frame_tier1_calls"], 0)
        self.assertIn(
            "majit_gc::standalone_alloc_nursery_collecting_typed_rooted",
            got["unmatched_seeds"],
        )

    def test_module_is_a_subject(self) -> None:
        self.assertTrue(any("pyre-module" in path for path in CHECKER.SUBJECTS))


if __name__ == "__main__":
    unittest.main()
