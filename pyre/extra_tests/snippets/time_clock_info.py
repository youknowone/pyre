import time


for name in ("time", "monotonic", "perf_counter", "process_time"):
    info = time.get_clock_info(name)
    assert isinstance(info.implementation, str)
    assert info.implementation
    assert isinstance(info.monotonic, bool)
    assert isinstance(info.adjustable, bool)
    assert isinstance(info.resolution, float)
    assert 0.0 < info.resolution <= 1.0

assert not time.get_clock_info("time").monotonic
assert time.get_clock_info("time").adjustable
assert time.get_clock_info("monotonic").monotonic
assert not time.get_clock_info("monotonic").adjustable

try:
    time.get_clock_info("unknown")
except ValueError as exc:
    assert str(exc) == "unknown clock"
else:
    raise AssertionError("unknown clock name was accepted")
