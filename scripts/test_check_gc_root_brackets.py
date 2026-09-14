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
    def test_skipped_liveness_is_a_shape_error(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            CHECKER.parse(SKIPPED_LIVENESS)
        self.assertIn("unbracketed_calls", str(raised.exception))

    def test_module_is_not_a_subject(self) -> None:
        self.assertNotIn("pyre-module", CHECKER.SUBJECT)
        self.assertTrue(CHECKER.SUBJECT.endswith("pyre-interpreter.ullbc"))


if __name__ == "__main__":
    unittest.main()
