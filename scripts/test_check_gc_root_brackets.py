from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("check-gc-root-brackets.py")
SPEC = importlib.util.spec_from_file_location("check_gc_root_brackets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

# After `register_module` supplies a PyObjectRef id, liveness prints
# but the crate has no PyFrame, so the frame scan is omitted.
MODULE_NO_FRAME = """\
== pyre_module (290 fun_decls) ==
   collecting-alloc seeds: 10 seed function(s); UNMATCHED patterns: ["majit_gc::standalone_alloc_nursery_collecting_typed_rooted"]
        cannot reach any collection        : 0
   unbracketed calls that can collect with a live PyObjectRef: 1 in 1 fn
       tier 1 (callee IS a dispatch seed): 0 call(s) in 0 fn
       tier 1.5 (live ptr later addressed as list/dict): 0 call(s) in 0 fn
   (no PyFrame pointer type id found — frame scan skipped)
"""

SKIPPED_LIVENESS = """\
== pyre_module (282 fun_decls) ==
   collecting-alloc seeds: 10 seed function(s); UNMATCHED patterns: ["x"]
        cannot reach any collection        : 0
    (no PyObjectRef type id found — liveness scan skipped)
"""


class GcRootParseTests(unittest.TestCase):
    def test_module_without_frames_is_zero_frames_not_a_shape_error(self) -> None:
        got = CHECKER.parse(MODULE_NO_FRAME)
        self.assertEqual(got["unbracketed_calls"], 1)
        self.assertEqual(got["frames_across_collecting"], 0)
        self.assertEqual(got["frame_tier1_calls"], 0)

    def test_skipped_liveness_is_still_a_shape_error(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            CHECKER.parse(SKIPPED_LIVENESS)
        self.assertIn("unbracketed_calls", str(raised.exception))

    def test_module_is_a_subject(self) -> None:
        self.assertTrue(any("pyre-module" in path for path in CHECKER.SUBJECTS))


if __name__ == "__main__":
    unittest.main()
