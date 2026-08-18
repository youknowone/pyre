import import_mutual1
import import_target
import import_target as aliased
from import_star import *
from import_target import func, other_func
from import_target import func as aliased_func
from import_target import other_func as aliased_other_func

assert import_target.X == import_target.func()
assert import_target.X == func()

assert import_mutual1.__name__ == "import_mutual1"

assert import_target.Y == other_func()

assert import_target.X == aliased.X
assert import_target.Y == aliased.Y

assert import_target.X == aliased_func()
assert import_target.Y == aliased_other_func()

assert STAR_IMPORT == "123"

try:
    from import_target import func, unknown_name

    raise AssertionError("`unknown_name` does not cause an exception")
except ImportError:
    pass

try:
    import mymodule
except ModuleNotFoundError as exc:
    assert exc.name == "mymodule"


test = __import__("import_target")
assert test.X == import_target.X

import builtins


class OverrideImportContext:
    def __enter__(self):
        self.original_import = builtins.__import__

    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.__import__ = self.original_import


with OverrideImportContext():

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        return len(name)

    builtins.__import__ = fake_import
    import test

    assert test == 4


# TODO: Once we can determine current directory, use that to construct this
# path:
# import sys
# sys.path.append("snippets/import_directory")
# import nested_target

# try:
#    X
# except NameError:
#    pass
# else:
#    raise AssertionError('X should not be imported')

from testutils import assert_raises

with assert_raises(SyntaxError):
    exec("import")


# interp_import.py:85-90 hands a cached package's fromlist to importlib's
# `_handle_fromlist` rather than re-entering `__import__`.  Pyre's slow path is
# the Python importlib bootstrap, so replacing that entry point makes an
# accidental fall through observable without depending on timing.
import importlib
import importlib._bootstrap as bootstrap

bootstrap_import = bootstrap.__import__
bootstrap_calls = []


def counting_bootstrap_import(*args, **kwargs):
    bootstrap_calls.append(args[0])
    return bootstrap_import(*args, **kwargs)


bootstrap.__import__ = counting_bootstrap_import
try:
    from importlib import import_module as cached_import_module

    assert cached_import_module is importlib.import_module
    assert bootstrap_calls == []
finally:
    bootstrap.__import__ = bootstrap_import
