import os


# The implementation exercised here is the POSIX `fcntl(FD_CLOEXEC)` port.
# Windows pyre currently uses the non-fd pathname-backed FileIO carrier, so it
# has no descriptor whose inheritance flag this fixture could inspect.
if os.name != "posix":
    print("OK")
    raise SystemExit


import tempfile  # noqa: E402


fd, path = tempfile.mkstemp()
os.close(fd)
try:
    with open(path, "rb") as stream:
        assert os.get_inheritable(stream.fileno()) is False

    def opener(name, flags):
        return os.open(name, flags)

    with open(path, "rb", opener=opener) as stream:
        assert os.get_inheritable(stream.fileno()) is False

    supplied = os.open(path, os.O_RDONLY)
    try:
        os.set_inheritable(supplied, True)
        with open(supplied, "rb", closefd=False) as stream:
            assert os.get_inheritable(stream.fileno()) is True
    finally:
        os.close(supplied)
finally:
    os.unlink(path)

print("OK")
