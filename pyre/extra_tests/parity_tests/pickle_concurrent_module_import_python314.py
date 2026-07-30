"""CPython 3.14 unpicklers wait for a concurrently initializing module."""

import os
import pickle
import shutil
import sys
import tempfile
import threading


root = tempfile.mkdtemp()
try:
    with open(os.path.join(root, "pickle_race_locker.py"), "w") as stream:
        stream.write("import threading\nbarrier = threading.Barrier(2)\n")
    with open(os.path.join(root, "pickle_race_target.py"), "w") as stream:
        stream.write(
            "import pickle_race_locker\n"
            "pickle_race_locker.barrier.wait(timeout=5)\n"
            "class Value:\n"
            "    pass\n"
        )

    sys.path.insert(0, root)
    import pickle_race_locker

    payload = b"\x80\x03cpickle_race_target\nValue\nq\x00)\x81q\x01."
    start = threading.Barrier(3)
    results = []
    errors = []

    def load():
        try:
            start.wait(timeout=5)
            results.append(pickle.loads(payload))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=load) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    pickle_race_locker.barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)

    assert not errors, errors
    assert all(not thread.is_alive() for thread in threads)
    from pickle_race_target import Value

    assert [type(result) for result in results] == [Value, Value]
finally:
    sys.path.remove(root)
    sys.modules.pop("pickle_race_locker", None)
    sys.modules.pop("pickle_race_target", None)
    shutil.rmtree(root)

print("OK")
