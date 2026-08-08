import _imp
import sys
from test.support import import_helper


name = "errno"
sys.modules.pop(name, None)

spec = type("Spec", (), {"name": name})()
module = _imp.create_builtin(spec)
assert module.__loader__ is None
assert module.__spec__ is None

sys.modules.pop(name, None)
bootstrap = import_helper.import_fresh_module(
    "importlib._bootstrap",
    fresh=("importlib",),
    blocked=("_frozen_importlib", "_frozen_importlib_external"),
)
loader = bootstrap.BuiltinImporter
module = loader.load_module(name)
assert module.__loader__ is loader
assert module.__spec__.loader is loader

import errno

assert errno.__loader__ is errno.__spec__.loader
assert errno.__loader__ is loader
assert errno.__name__ == name
assert errno.__package__ == ""
assert sys.modules[name] is errno

print("OK")
