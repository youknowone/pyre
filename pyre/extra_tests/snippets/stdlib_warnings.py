import _warnings
import sys
import warnings


expected = [
    ("default", None, DeprecationWarning, "__main__", 0),
    ("ignore", None, DeprecationWarning, None, 0),
    ("ignore", None, PendingDeprecationWarning, None, 0),
    ("ignore", None, ImportWarning, None, 0),
    ("ignore", None, ResourceWarning, None, 0),
]
assert _warnings.filters == expected
assert _warnings._onceregistry == {}
assert _warnings._defaultaction == "default"

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    source = []
    _warnings.warn("native warning", UserWarning, source=source)
    warning = caught[-1]
    assert str(warning.message) == "native warning"
    assert warning.category is UserWarning
    assert warning.filename == __file__
    assert warning.source is source
    assert warning.line is None

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("ignore", UserWarning)
    registry = {}
    _warnings.warn_explicit(
        "ignored", UserWarning, "<warnings-test>", 44, registry=registry
    )
    assert caught == []
    assert list(registry) == ["version"]

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("once")
    registry = {}
    for _ in range(3):
        _warnings.warn_explicit(
            "only once", UserWarning, "<warnings-test>", 12, registry=registry
        )
    assert len(caught) == 1

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("all")
    _warnings.warn_explicit("all alias", UserWarning, "<warnings-test>", 1)
    _warnings.warn_explicit("all alias", UserWarning, "<warnings-test>", 1)
    assert len(caught) == 2

try:
    _warnings.warn("bad category", 123)
except TypeError:
    pass
else:
    raise AssertionError("non-Warning category accepted")

for kwargs in ({"module_globals": True}, {"registry": 42}):
    try:
        _warnings.warn_explicit("bad mapping", UserWarning, "filename", 1, **kwargs)
    except (TypeError, AttributeError):
        pass
    else:
        raise AssertionError("invalid warning mapping accepted")


class Loader:
    def get_source(self, fullname):
        assert fullname == "warnings_source_test"
        return "first line\nsecond line\n"


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _warnings.warn_explicit(
        "loader source",
        UserWarning,
        "loader.py",
        2,
        module_globals={"__loader__": Loader(), "__name__": "warnings_source_test"},
    )
    assert len(caught) == 1
    # PyPy uses a loader-provided source line only for the direct stderr
    # fallback; WarningMessage.line remains None.
    assert caught[0].line is None


class BadLoader:
    def get_source(self, fullname):
        class BadSource(str):
            def splitlines(self):
                return 42

        return BadSource("source")


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _warnings.warn_explicit(
        "bad loader source",
        UserWarning,
        "loader.py",
        1,
        module_globals={"__loader__": BadLoader(), "__name__": "bad_source"},
    )
    assert len(caught) == 1

print("stdlib warnings ok", sys.version_info[:2])
