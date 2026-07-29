"""CPython 3.14 singleton-type pickle parity."""

import pickle


for singleton in (None, Ellipsis, NotImplemented):
    singleton_type = type(singleton)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(singleton_type, protocol))
        assert restored is singleton_type, (singleton_type, protocol, restored)

print("OK")
