# pyre-check: no-cpython

import array
import contextvars
import gc
import json
import pickle
import re
import sys
import types
import unicodedata
import weakref
from collections import deque


results = []


def managed(label, value):
    results.append((label, value))
    return value


# _json: encoder/decoder results, error notes, one-shot chunks, and float keys.
json_source = "gc-json-probe-" + ("x" * 37)
managed("json encode", json.encoder.encode_basestring(json_source))
json_decoded, json_end = json.decoder.scanstring('"' + json_source + '"', 1)
assert json_decoded == json_source
assert json_end == len(json_source) + 2
managed("json decode", json_decoded)

try:
    json.dumps([object()])
except TypeError as exc:
    if not getattr(exc, "__notes__", None):
        exc.add_note("gc-json-note-" + ("x" * 37))
    managed("json note", exc.__notes__[-1])
else:
    raise AssertionError("json.dumps accepted an unsupported object")

json_chunks = list(json.JSONEncoder().iterencode({"runtime": json_source}, _one_shot=True))
assert json_chunks
if json.encoder.c_make_encoder is not None:
    managed("json chunk", json_chunks[0])

float_key = 1.2345678901234567e123
float_keys = []


def observe_float_key(key):
    if key.startswith("1.234567890123456") and key.endswith("e+123"):
        float_keys.append(key)
    return json.encoder.encode_basestring_ascii(key)


if json.encoder.c_make_encoder is None:
    managed("json float key oracle", repr(float_key))
else:
    encoder = json.encoder.c_make_encoder(
        {}, lambda obj: None, observe_float_key, None, ": ", ", ", False, False, True
    )
    assert encoder({float_key: None}, 0) == ['{"1.2345678901234567e+123": null}']
    assert len(float_keys) == 1
    managed("json float key", float_keys[0])


# Formatted repr paths whose ordinary repr formatting is covered elsewhere.
alias_name = "RuntimeAlias" + ("X" * 19)
alias_type = type(alias_name, (), {})
managed("GenericAlias repr", types.GenericAlias.__repr__(list[alias_type]))

context_name = "runtime-context-" + ("x" * 29)
context_var = contextvars.ContextVar(context_name)
managed("ContextVar repr", repr(context_var))
context_token = context_var.set(object())
managed("ContextVar token repr", repr(context_token))

managed("structseq repr", repr(sys.version_info))
managed("array tounicode", array.array("u", "abc").tounicode())
managed("array int repr", array.array.__repr__(array.array("i", [1, 2, 3])))
managed("array unicode repr", array.array.__repr__(array.array("u", "abc")))
managed("deque repr", deque.__repr__(deque([1, 2, 3], maxlen=4)))


# _sre: final repr/substitution strings and every slice-container assembly shape.
pattern = re.compile(r"(?P<word>[a-z]+)-(?P<number>\d+)")
subject = "alpha-123-omega"
match = pattern.search(subject)
managed("SRE pattern repr", type(pattern).__repr__(pattern))
managed("SRE match repr", type(match).__repr__(match))
managed("SRE sub", pattern.sub("word", subject))
subn_result, subn_count = pattern.subn("word", subject)
assert subn_count == 1
managed("SRE subn", subn_result)
managed("SRE expand", match.expand(r"<\1:\2>"))

managed("SRE group", match.group(1))
managed("SRE getitem", match[2])
managed("SRE multi-group", match.group(1, 2)[0])
managed("SRE groups", match.groups()[1])
managed("SRE groupdict", match.groupdict()["word"])
managed("SRE findall plain", re.findall(r"[a-z]+", subject)[0])
managed("SRE findall single", re.findall(r"([a-z]+)", subject)[1])
managed("SRE findall multiple", pattern.findall(subject)[0][1])
split_parts = re.split(r"(-)", subject)
managed("SRE split text", split_parts[0])
managed("SRE split group", split_parts[1])


class Text(str):
    pass


unchanged = pattern.sub("replacement", Text("no match here"))
assert type(unchanged) is str
managed("SRE subclass normalization", unchanged)


# _pickle's three Unicode opcodes share one loader but have distinct lengths.
pickle_text = b"pickle runtime string"
pickle_payloads = (
    b"\x80\x04\x8c" + bytes([len(pickle_text)]) + pickle_text + b".",
    b"\x80\x04X" + len(pickle_text).to_bytes(4, "little") + pickle_text + b".",
    b"\x80\x04\x8d" + len(pickle_text).to_bytes(8, "little") + pickle_text + b".",
)
pickle_opcodes = ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8")
for opcode, payload in zip(pickle_opcodes, pickle_payloads):
    managed(opcode, pickle.loads(payload))


# normalize has separate non-ASCII, exact-ASCII identity, and subclass paths.
for form, source, expected in (
    ("NFC", "e\u0301 runtime", "é runtime"),
    ("NFD", "é runtime", "e\u0301 runtime"),
    ("NFKC", "\ufb03 runtime", "ffi runtime"),
    ("NFKD", "\ufb03 runtime", "ffi runtime"),
):
    normalized = unicodedata.normalize(form, source)
    assert normalized == expected
    managed("normalize " + form, normalized)

ascii_source = "".join(["ascii", " runtime"])
ascii_result = unicodedata.normalize("NFC", ascii_source)
assert ascii_result is ascii_source
managed("normalize ASCII identity", ascii_result)
subclass_result = unicodedata.normalize("NFC", Text("subclass runtime"))
assert type(subclass_result) is str
managed("normalize subclass", subclass_result)


# SimpleNamespace's recursive guard is a separate result branch.
namespace = types.SimpleNamespace(alpha="value", count=3)
managed("SimpleNamespace repr", types.SimpleNamespace.__repr__(namespace))
recursive_result = None


class CaptureRecursiveRepr:
    def __repr__(self):
        global recursive_result
        recursive_result = repr(recursive_namespace)
        return "captured"


recursive_namespace = types.SimpleNamespace(value=CaptureRecursiveRepr())
assert repr(recursive_namespace) == "namespace(value=captured)"
assert recursive_result == "namespace(...)"
managed("SimpleNamespace recursive repr", recursive_result)


# BaseException rendering has arity/identity/specialized branches.
for label, exception in (
    ("exception repr empty", ValueError()),
    ("exception repr single", ValueError("x")),
    ("exception repr multiple", ValueError("x", 1)),
):
    managed(label, BaseException.__repr__(exception))


class Render:
    def __init__(self, result):
        self.result = result

    def __str__(self):
        return self.result


runtime_text = "runtime-exception-value-" + str(id(gc))
rendered_text = "rendered-exception-value-" + str(id(runtime_text))
assert BaseException.__str__(ValueError(runtime_text)) is runtime_text
managed("exception str identity", runtime_text)
assert BaseException.__str__(ValueError(Render(rendered_text))) is rendered_text
managed("exception str delegated identity", rendered_text)
managed("exception str multiple", BaseException.__str__(ValueError(runtime_text, 1)))
managed("KeyError str", KeyError.__str__(KeyError(runtime_text)))

group_message = "runtime-exception-group-message-" + str(id(results))
group = ExceptionGroup(group_message, [ValueError("runtime group leaf")])
managed("ExceptionGroup str", ExceptionGroup.__str__(group))
managed("ExceptionGroup repr", ExceptionGroup.__repr__(group))


# Direct descriptor paths share typedef formatting but have distinct owners.
class RuntimeOwner:
    def runtime_method(self):
        pass


owner = RuntimeOwner()
managed("bound method repr", type(owner.runtime_method).__repr__(owner.runtime_method))
managed("super repr", super.__repr__(super(RuntimeOwner, owner)))
union = int | str
managed("union repr", type(union).__repr__(union))


def runtime_function():
    pass


managed("function repr", type(runtime_function).__repr__(runtime_function))
managed("builtin function repr", type(len).__repr__(len))
managed("builtin method repr", type([].append).__repr__([].append))
wrapper = (1).__add__
managed("method-wrapper repr", type(wrapper).__repr__(wrapper))


class RuntimeType:
    pass


managed("type repr", type.__repr__(RuntimeType))


def make_cell(value):
    def inner():
        return value

    return inner.__closure__[0]


filled_cell = make_cell(42)
empty_cell = make_cell(42)
del empty_cell.cell_contents
managed("filled cell repr", type(filled_cell).__repr__(filled_cell))
managed("empty cell repr", type(empty_cell).__repr__(empty_cell))


def generate():
    yield 1


generator = generate()
managed("generator repr", type(generator).__repr__(generator))
generator.close()


async def run():
    return 1


coroutine = run()
managed("coroutine repr", type(coroutine).__repr__(coroutine))
coroutine.close()


class SlotOwner:
    __slots__ = ("value",)


for label, descriptor in (
    ("getset descriptor repr", type.__dict__["__name__"]),
    ("member descriptor repr", SlotOwner.__dict__["value"]),
    ("method descriptor repr", list.__dict__["append"]),
    ("classmethod descriptor repr", dict.__dict__["fromkeys"]),
):
    managed(label, type(descriptor).__repr__(descriptor))


# Stateful reprs have distinct live/released and live/dead branches.
view = memoryview(b"x")
managed("memoryview live repr", memoryview.__repr__(view))
view.release()
managed("memoryview released repr", memoryview.__repr__(view))

weak_marker = "".join(["dynamic", " weak proxy"])


class Referent:
    def __str__(self):
        return weak_marker


class CallableReferent:
    def __call__(self):
        pass


referent = Referent()
callable_referent = CallableReferent()
reference = weakref.ref(referent)
proxy = weakref.proxy(referent)
callable_proxy = weakref.proxy(callable_referent)
proxy_text = type(proxy).__str__(proxy)
assert proxy_text is weak_marker
managed("weak proxy str identity", proxy_text)
managed("weakref live repr", weakref.ReferenceType.__repr__(reference))
managed("weak proxy live repr", weakref.ProxyType.__repr__(proxy))
managed("callable weak proxy live repr", weakref.CallableProxyType.__repr__(callable_proxy))

del referent
del callable_referent
gc.collect()
for label, value in (
    ("weakref dead repr", weakref.ReferenceType.__repr__(reference)),
    ("weak proxy dead repr", weakref.ProxyType.__repr__(proxy)),
    ("callable weak proxy dead repr", weakref.CallableProxyType.__repr__(callable_proxy)),
):
    assert "; dead>" in value
    managed(label, value)


# Take one heap census after every result is rooted in `results`.  The census
# names only tracked types and a string is never one, so it can no longer
# report a result directly.  It does name `results`, and the collector's own
# edge walk goes the rest of the way: one hop reaches the entry tuples and a
# second reaches the strings.  A value the runtime handed back without giving
# it a managed identity would be missing from that walk.
assert any(obj is results for obj in gc.get_objects())
reached = []
for entry in gc.get_referents(results):
    reached.extend(gc.get_referents(entry))
for label, value in results:
    assert any(ref is value for ref in reached), label

print("runtime string results are collectable")
