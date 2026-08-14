# CPython-suite gap: pickle tests do not keep dump hooks live across collection.
# parity-tests reason: this guards pyre/PyPy moving-GC roots during pickling.

"""Pickler hooks and the effective dispatch table survive collection."""

import copyreg
import gc
import io
import pickle


class Payload:
    def __init__(self, value):
        self.value = value


def rebuild_payload(value):
    return Payload(value)


def reduce_payload(obj):
    return rebuild_payload, (obj.value,)


class CollectingPickler(pickle.Pickler):
    def __init__(self, file, protocol):
        super().__init__(file, protocol=protocol)
        self.churned = False
        self.persistent_before_churn = 0
        self.persistent_after_churn = 0
        self.override_after_churn = 0

    def persistent_id(self, obj):
        if self.churned:
            self.persistent_after_churn += 1
        else:
            self.persistent_before_churn += 1
            junk = [[object() for _ in range(64)] for _ in range(128)]
            assert len(junk) == 128
            gc.collect()
            self.churned = True
        return None

    def reducer_override(self, obj):
        assert self.churned
        self.override_after_churn += 1
        return NotImplemented


original_dispatch_table = copyreg.dispatch_table
try:
    copyreg.dispatch_table = {Payload: reduce_payload}
    expected = [Payload(i) for i in range(40)]
    expected.extend([{"padding": [str(i) for i in range(80)]}, Payload(999)])

    for protocol in (0, 2, pickle.HIGHEST_PROTOCOL):
        stream = io.BytesIO()
        pickler = CollectingPickler(stream, protocol)
        pickler.dump(expected)
        restored = pickle.loads(stream.getvalue())

        assert [obj.value for obj in restored[:40]] == list(range(40))
        assert restored[-1].value == 999
        assert restored[-2] == expected[-2]
        assert pickler.persistent_before_churn == 1
        assert pickler.persistent_after_churn > 0
        assert pickler.override_after_churn >= 41
finally:
    copyreg.dispatch_table = original_dispatch_table

print("OK")
