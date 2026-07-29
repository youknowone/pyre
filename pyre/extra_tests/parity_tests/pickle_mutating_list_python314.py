"""CPython 3.14 behavior when pickling a list that clears itself."""

import io
import pickle


class Clearer:
    pass


for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    collection = [Clearer(), Clearer()]

    class EvilPickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Clearer):
                collection.clear()
            return None

    try:
        EvilPickler(io.BytesIO(), protocol=protocol).dump(collection)
    except RuntimeError as exc:
        assert str(exc) == "list changed size during iteration"

print("OK")
