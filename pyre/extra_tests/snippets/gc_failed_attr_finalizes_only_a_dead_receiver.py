# pyre-check: gate=1
"""A failed LOAD_ATTR runs a user finalizer only for a dead temporary.

CPython drops the popped receiver immediately.  pyre approximates that with a
collect after POP_EXCEPT, the same boundary `test_io.test_error_through_destructor`
uses via `assertRaises`.  Arm only when this object is registered for a
finalizer.  A live `f.missing` stays reachable through the local, so its
`__del__` must not run; `C().missing` must.
"""

import unittest

seen = []


class Watched:
    def __init__(self, name):
        self.name = name

    def __del__(self):
        seen.append(self.name)


class FailedAttrFinalizer(unittest.TestCase):
    def test_only_the_dead_receiver_is_finalized(self):
        live = Watched("live")
        with self.assertRaises(AttributeError):
            live.missing
        self.assertEqual(seen, [])
        with self.assertRaises(AttributeError):
            Watched("temp").missing
        self.assertEqual(seen, ["temp"])
        self.assertEqual(live.name, "live")


FailedAttrFinalizer().test_only_the_dead_receiver_is_finalized()
