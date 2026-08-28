# CPython-suite gap: `test_pickle` reaches this, but only as
# `PyPicklingErrorTests.test_wrong_object_lookup_error` failing an
# `assertIsNone(cm.exception.__context__)` several hundred tests after the load
# that poisoned the state, so the suite reports it far from its cause.
# parity-tests reason: `_Unpickler.load` returns from inside `except _Stop`, so
# every load runs PUSH_EXC_INFO then POP_EXCEPT.  Once a bridge resumes into
# that handler, PUSH_EXC_INFO's `prev` save read the exception the bridge was
# resuming with instead of the execution context's slot, and the matching
# POP_EXCEPT reinstated the just-handled `_Stop` as the thread's current
# exception -- so the NEXT unrelated raise, anywhere, chained it as
# `__context__`.

import pickle


def current_context():
    """The exception a fresh raise would chain, i.e. the handled-exception slot."""
    try:
        raise ValueError("probe")
    except ValueError as exc:
        return exc.__context__


payload = pickle._dumps(set(range(10)))

for iteration in range(1, 3001):
    loaded = pickle._loads(payload)
    assert loaded == set(range(10)), (iteration, loaded)
    context = current_context()
    assert context is None, (iteration, repr(context))

print("OK")
