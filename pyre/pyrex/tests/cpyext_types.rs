//! End-to-end check for C-defined types: `PyType_Ready`, the slot wrappers,
//! `tp_methods`/`tp_members`/`tp_getset` descriptors, inheritance through
//! `tp_base` and `PyErr_NewExceptionWithDoc`.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import cpyext_types as m

# ── the type itself ────────────────────────────────────────────────────
assert m.Point.__name__ == 'Point'
assert m.Point.__module__ == 'cpyext_types'
assert repr(m.Point.__dict__['norm']) == "<method 'norm' of 'cpyext_types.Point' objects>"
assert m.Point.__doc__ == 'a two-dimensional point defined in C'
assert isinstance(m.Point, type)
assert m.flags() == (1, 1)

# A C type whose base is `int` earns the long fast-subclass flag; one based on
# `object` earns none of them.
made, is_long, is_unicode, point_is_long = m.subclass_flags()
assert issubclass(made, int), made
assert (is_long, is_unicode, point_is_long) == (1, 0, 0), \
    (is_long, is_unicode, point_is_long)

# ── tp_new and tp_init ─────────────────────────────────────────────────
p = m.Point()
assert (p.x, p.y) == (0, 0)
assert p.label == ''
assert p.scale == 1.0

p = m.Point(3, 4)
assert (p.x, p.y) == (3, 4)
p = m.Point(3, 4, 'origin')
assert p.label == 'origin'
p = m.Point(y=9, x=1, label='kw')
assert (p.x, p.y, p.label) == (1, 9, 'kw')

try:
    m.Point(1, 2, 3, 4)
except TypeError:
    pass
else:
    raise AssertionError('too many arguments accepted')

# ── tp_members ─────────────────────────────────────────────────────────
p = m.Point(3, 4)
p.x = 10
assert p.x == 10
p.scale = 2.5
assert p.scale == 2.5
p.label = ['not', 'a', 'string']
assert p.label == ['not', 'a', 'string']

try:
    p.y = 1
except AttributeError:
    pass
else:
    raise AssertionError('a READONLY member was written')

assert type(m.Point.__dict__['x']).__name__ == 'member_descriptor'
assert repr(m.Point.__dict__['x']) == "<member 'x' of 'cpyext_types.Point' objects>"
assert m.Point.__dict__['x'].__doc__ == 'the abscissa'
assert m.Point.__dict__['x'].__name__ == 'x'
assert m.Point.__dict__['x'].__objclass__ is m.Point

# ── tp_getset ──────────────────────────────────────────────────────────
p = m.Point(3, 4)
assert p.total == 3 + 4 + 100
p.total = 12
assert (p.x, p.y) == (12, 0)
assert p.frozen == 'frozen'
try:
    p.frozen = 'thawed'
except AttributeError:
    pass
else:
    raise AssertionError('a get-only property was written')

assert type(m.Point.__dict__['total']).__name__ == 'getset_descriptor'
assert repr(m.Point.__dict__['total']) == "<attribute 'total' of 'cpyext_types.Point' objects>"
assert m.Point.__dict__['total'].__doc__ == 'x + y + closure'
assert m.Point.__dict__['total'].__objclass__ is m.Point

# ── tp_methods ─────────────────────────────────────────────────────────
p = m.Point(3, 4)
assert p.norm() == 25
assert p.translate(1, 1) is p
assert (p.x, p.y) == (4, 5)
assert p.named() == 'p:4'
assert p.named('q') == 'q:4'
assert p.named(prefix='r') == 'r:4'
try:
    p.norm(1)
except TypeError:
    pass
else:
    raise AssertionError('METH_NOARGS took an argument')

assert type(m.Point.__dict__['norm']).__name__ == 'method_descriptor'
assert m.Point.__dict__['norm'].__doc__ == 'squared length'
assert m.Point.__dict__['norm'].__objclass__ is m.Point
# An unbound descriptor takes the receiver as its first argument.
assert m.Point.__dict__['norm'](p) == 4 * 4 + 5 * 5

# ── tp_repr, tp_str, tp_hash, tp_call ──────────────────────────────────
p = m.Point(3, 4)
assert repr(p) == 'Point(3, 4)'
assert str(p) == '3/4'
assert hash(p) == 3 * 1000003 + 4
assert p(1, 2) == (4, 6)
assert p(dy=10) == (3, 14)
assert p() == (3, 4)

# ── tp_richcompare ─────────────────────────────────────────────────────
small = m.Point(1, 1)
large = m.Point(5, 5)
assert small < large
assert small <= large
assert large > small
assert large >= small
assert small != large
assert small == m.Point(1, 1)
assert not (small == large)
assert (small == 'not a point') is False

# ── tp_base inheritance ────────────────────────────────────────────────
assert issubclass(m.Point3, m.Point)
assert m.is_subtype() == (1, 0)
q = m.Point3(2, 3)
assert isinstance(q, m.Point)
assert (q.x, q.y) == (2, 3)
assert q.z == 0
q.z = 7
assert q.depth() == 7
# Inherited slots: Point3 declares none of these.
assert repr(q) == 'Point(2, 3)'
assert q.norm() == 13
assert q.total == 2 + 3 + 100

# ── tp_iter and tp_iternext ────────────────────────────────────────────
assert list(m.Counter(4)) == [0, 1, 2, 3]
counter = m.Counter(2)
assert iter(counter) is counter
assert next(counter) == 0
assert next(counter) == 1
try:
    next(counter)
except StopIteration:
    pass
else:
    raise AssertionError('the exhausted iterator kept going')

# ── the C side reads its own instances back ────────────────────────────
made = m.make(6, 7)
assert type(made) is m.Point
assert (made.x, made.y, made.label) == (6, 7, 'made')
assert m.is_point(made) is True
assert m.is_point(m.Point3(0, 0)) is True
assert m.is_point(object()) is False
assert m.sum_x(made) == 6

# ── PyErr_NewExceptionWithDoc ──────────────────────────────────────────
assert issubclass(m.TypesError, Exception)
assert m.TypesError.__name__ == 'TypesError'
assert m.TypesError.__module__ == 'cpyext_types'
assert m.TypesError.__doc__ == 'raised by the fixture'
try:
    m.sum_x(object())
except m.TypesError as error:
    assert str(error) == 'not a Point', str(error)
else:
    raise AssertionError('the module exception did not propagate')

# ── the number table ───────────────────────────────────────────────────
v = m.Vec(3)
assert repr(v) == 'Vec(3)'
assert v.value == 3
assert repr(v + m.Vec(4)) == 'Vec(7)'
assert repr(v + 10) == 'Vec(13)'
assert repr(10 + v) == 'Vec(13)'
assert repr(v - 1) == 'Vec(2)'
assert repr(1 - v) == 'Vec(-2)'
assert repr(v * 4) == 'Vec(12)'
assert repr(4 * v) == 'Vec(12)'
assert repr(-v) == 'Vec(-3)'
assert repr(abs(m.Vec(-9))) == 'Vec(9)'
assert repr(v ** 3) == 'Vec(27)'
assert repr(pow(m.Vec(2), 10, 1000)) == 'Vec(24)'
assert int(v) == 3
assert float(v) == 3.0
assert bool(v) is True
assert bool(m.Vec(0)) is False
try:
    v + 'text'
except TypeError:
    pass
else:
    raise AssertionError('an unsupported operand was accepted')

acc = m.Vec(1)
same = acc
acc += 5
assert acc is same
assert acc.value == 6

# ── the sequence table ─────────────────────────────────────────────────
bag = m.Bag(4, 5, 6)
assert len(bag) == 3
assert bag[0] == 4
assert bag[-1] == 6
assert 5 in bag
assert 9 not in bag
bag[1] = 50
assert bag[1] == 50
del bag[0]
assert len(bag) == 2
assert bag[0] == 50
try:
    bag[9]
except IndexError:
    pass
else:
    raise AssertionError('an out-of-range index was accepted')
assert bag * 2 == [50, 6, 50, 6]
assert 2 * bag == [50, 6, 50, 6]
assert list(bag) == [50, 6]

# ── the mapping table ──────────────────────────────────────────────────
table = m.Table()
assert len(table) == 0
table['a'] = 1
table['b'] = 2
assert len(table) == 2
assert table['a'] == 1
try:
    table['missing']
except KeyError:
    pass
else:
    raise AssertionError('a missing key was accepted')
del table['a']
assert len(table) == 1
assert sorted(table.keys()) == ['b']

# ── the abstract protocols, driven from C ──────────────────────────────
p = m.protocol
assert p('add', 2, 3) == 5
assert p('add', 'a', 'b') == 'ab'
assert p('add', [1], [2]) == [1, 2]
assert repr(p('add', m.Vec(1), m.Vec(2))) == 'Vec(3)'
assert p('multiply', 3, 4) == 12
assert p('power', 2, 8) == 256
assert p('negative', 5) == -5
assert p('index', 7) == 7
assert p('float', 7) == 7.0
assert p('number_check', 7) is True
assert p('number_check', 'x') is False
assert p('as_ssize', 12) == 12

assert p('sequence_check', [1, 2]) is True
assert p('sequence_check', 5) is False
assert p('size', [1, 2, 3]) == 3
assert p('size', bag) == 2
assert p('getitem', [1, 2, 3], 1) == 2
assert p('getitem', bag, 0) == 50
assert p('contains', [1, 2, 3], 2) is True
assert p('contains', bag, 6) is True
assert p('list', (1, 2)) == [1, 2]
assert p('tuple', [1, 2]) == (1, 2)
assert p('seq_index', [7, 8, 9], 8) == 1
assert p('repeat', [1, 2], 2) == [1, 2, 1, 2]

assert p('mapping_check', {'a': 1}) is True
assert p('mapping_check', [1]) is False
assert p('keys', {'a': 1}) == ['a']
assert p('values', {'a': 1}) == [1]
assert p('items', {'a': 1}) == [('a', 1)]
assert p('getstring', {'a': 1}, 'a') == 1
assert p('haskey', {'a': 1}, 'a') is True
assert p('haskey', {'a': 1}, 'z') is False

# ── PyType_FromSpec ────────────────────────────────────────────────────
assert m.Spec.__name__ == 'Spec'
assert m.Spec.__doc__ == 'a heap type built from a spec'
assert m.Spec.__module__ == 'cpyext_types'
s = m.Spec(21)
assert repr(s) == 'Spec(21)'
assert s.code == 21
assert s.double() == 42
assert len(s) == 21
s.code = 4
assert s.double() == 8

# ── the module and the token a spec type carries ───────────────────────
module, by_def_is_module, module_name, qualified = m.type_owner(m.Spec)
assert module is m
assert by_def_is_module == 1
assert module_name == 'cpyext_types', module_name
assert qualified == 'cpyext_types.Spec', qualified

# Py_TP_USE_SPEC made the spec's own address the token, so Spec answers for
# itself and Extra answers for Spec through its base.
found, owner, extra_found, extra_owner, absent = m.type_token(m.Spec)
assert (found, owner) == (1, m.Spec), (found, owner)
assert (extra_found, extra_owner) == (0, None), (extra_found, extra_owner)
assert absent == 0

found, owner, extra_found, extra_owner, absent = m.type_token(m.Extra)
assert (found, owner) == (1, m.Spec), (found, owner)
assert (extra_found, extra_owner) == (1, m.Extra), (extra_found, extra_owner)
assert absent == 0

# A type built without a token matches none of them.
found, owner, extra_found, extra_owner, absent = m.type_token(m.Point)
assert (found, owner, extra_found, extra_owner, absent) == (0, None, 0, None, 0)

try:
    m.type_token_null(m.Spec)
except SystemError:
    pass
else:
    raise AssertionError('a NULL token was accepted')

# ── a spec declaring storage relative to its base's ────────────────────
assert issubclass(m.Extra, m.Spec)
assert m.type_data_size(m.Extra) >= 16, m.type_data_size(m.Extra)
# Spec declares a whole block, so it extends its base by nothing.
assert m.type_data_size(m.Spec) == 0, m.type_data_size(m.Spec)

e = m.Extra(9)
assert e.code == 9
e.set(3, 0.5)
assert e.get() == (3, 0.5, 9), e.get()
# The base's own storage is untouched by the extra data behind it.
e.code = 11
assert e.get() == (3, 0.5, 11), e.get()

module, by_def_is_module, module_name, qualified = m.type_owner(m.Extra)
assert module is m
assert qualified == 'cpyext_types.Extra', qualified

# A static type carries no module of its own.
try:
    m.type_owner(m.Point)
except TypeError:
    pass
else:
    raise AssertionError('a static type reported a module')

# ── PyType_Freeze ──────────────────────────────────────────────────────
class Mutable:
    pass

Mutable.before = 1
m.freeze(Mutable)
assert Mutable.before == 1
try:
    Mutable.after = 2
except TypeError:
    pass
else:
    raise AssertionError('a frozen class accepted an attribute')

# A class whose base is still mutable cannot be frozen.
class Derived(Mutable):
    pass

class Deeper(Derived):
    pass

try:
    m.freeze(Deeper)
except TypeError:
    pass
else:
    raise AssertionError('a class with a mutable base was frozen')

# ── tp_descr_get and tp_descr_set ──────────────────────────────────────
class Holder:
    field = m.Doubler()

h = Holder()
assert h.field == 0
h.field = 5
assert h.field == 10
del h.field
assert h.field == 0
# A class access hands the descriptor back.
assert type(Holder.field) is m.Doubler

# ── the buffer table ───────────────────────────────────────────────────
blob = m.Blob(b'abcdef')
assert blob.exports() == 0
view = memoryview(blob)
assert blob.exports() == 1
assert view.obj is blob
assert len(view) == 6
assert view.readonly is False
assert view.itemsize == 1
assert view.format == 'B'
assert bytes(view) == b'abcdef'
assert view[0] == ord('a')
assert list(view[1:3]) == [ord('b'), ord('c')]
view.release()
assert blob.exports() == 0

with memoryview(blob) as inner:
    assert inner[5] == ord('f')
    assert blob.exports() == 1
assert blob.exports() == 0

# The window is the exporter's own storage, not a copy of it.
live = m.Blob(b'abc')
with memoryview(live) as writable:
    writable[0] = ord('z')
assert live.read(live) == b'zbc'

# The bytes-like conversions reach `bf_getbuffer` too.
assert bytes(m.Blob(b'xy')) == b'xy'
assert bytearray(m.Blob(b'xy')) == bytearray(b'xy')

# PyObject_GetBuffer driven from C, over a C exporter and over a pyre object.
assert m.Blob(b'').read(m.Blob(b'held')) == b'held'
assert m.Blob(b'').read(b'plain bytes') == b'plain bytes'
assert m.Blob(b'').read(bytearray(b'mutable')) == b'mutable'
try:
    m.Blob(b'').read(42)
except TypeError:
    pass
else:
    raise AssertionError('a non-exporter was accepted')

# ── the async table ────────────────────────────────────────────────────
ticker = m.Ticker(3)
assert list(ticker.__await__()) == [3, 2, 1]
assert ticker.__aiter__() is ticker
assert ticker.__anext__() == 2
assert ticker.__anext__() == 1
assert ticker.__anext__() == 0
try:
    ticker.__anext__()
except StopAsyncIteration:
    pass
else:
    raise AssertionError('the exhausted async iterator kept going')

# `await` goes through `am_await`, whose iterator's yields reach the caller.
async def use(source):
    return await source

coroutine = use(m.Ticker(2))
assert coroutine.send(None) == 2
assert coroutine.send(None) == 1
try:
    coroutine.send(None)
except StopIteration as stop:
    assert stop.value is None, stop.value
else:
    raise AssertionError('the coroutine did not finish')

# `am_send`, which has no dunder of its own, is reached through PyIter_Send.
stepper = m.Ticker(2)
assert m.send(stepper) == ('next', 1)
assert m.send(stepper) == ('next', 0)
assert m.send(stepper) == ('return', -1)
# A pyre iterator goes through `__next__` / `send` instead.
assert m.send(iter([7, 8])) == ('next', 7)

# ── capsules ───────────────────────────────────────────────────────────
capsule = m.PAYLOAD
assert repr(capsule).startswith('<capsule object "cpyext_types.PAYLOAD" at 0x')
assert m.capsule_read(capsule) == 4242
assert m.capsule_facts(capsule) == ('cpyext_types.PAYLOAD', 1, 0, 1, 1)
assert m.capsule_import() == 4242
try:
    m.capsule_wrong_name(capsule)
except ValueError as error:
    assert 'incorrect name' in str(error), str(error)
else:
    raise AssertionError('a mismatched capsule name was accepted')
try:
    m.capsule_read(object())
except ValueError as error:
    assert 'invalid PyCapsule' in str(error), str(error)
else:
    raise AssertionError('a non-capsule was accepted')

# ── imports ────────────────────────────────────────────────────────────
import sys
assert m.import_attr('sys', 'maxsize') == sys.maxsize
assert m.import_attr('cpyext_types', 'ANSWER_TYPES') == 'types'
assert m.add_module_ref('sys') is sys
fresh = m.add_module_ref('cpyext_fresh_module')
assert fresh.__name__ == 'cpyext_fresh_module'
assert sys.modules['cpyext_fresh_module'] is fresh
try:
    m.import_attr('cpyext_no_such_module', 'x')
except ImportError:
    pass
else:
    raise AssertionError('a missing module imported')

print('cpyext-types-ok')
"#;

#[test]
fn defines_types_from_c() {
    let fixtures = Fixtures::new("cpyext-types");
    fixtures.compile("cpyext_types");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-types-ok");
}

/// Both halves of the `rawrefcount` P-link rule, neither of which any other
/// test observes: a mirror at exactly the link share does not root its object,
/// so the instance dies and its `tp_dealloc` runs; a mirror above the link
/// share does root it (`rawrefcount.rst`, "mark 'p' as surviving").
///
/// Every step needs a real collection between the drop and the observation,
/// and one more after it: the dead queue is drained by an async action that
/// runs *after* the collection that filled it, so the reference it releases is
/// only reclaimable by the next one.
///
/// A cycle running through a C reference is deliberately not asserted here.
/// It is not collectable — that needs `tp_traverse`/`tp_clear` participation,
/// which no `rawrefcount` collection consults, upstream included.
const LIFETIME_SCRIPT: &str = r#"
import gc
import weakref
import cpyext_types as m

# ── an instance nothing holds is deallocated ───────────────────────────
before = m.owner_deallocs()
for _ in range(16):
    m.Owner()
gc.collect()
after = m.owner_deallocs()
assert after > before, f'no Owner was deallocated: {before} -> {after}'

# ── a C reference roots its object ─────────────────────────────────────
class Node:
    pass

node = Node()
ref = weakref.ref(node)
owner = m.Owner(node)
del node
gc.collect()
gc.collect()
assert ref() is not None, 'a C reference did not keep its object alive'

# ── and releasing it lets the object go ────────────────────────────────
before = m.owner_deallocs()
del owner
gc.collect()
after = m.owner_deallocs()
assert after > before, f'the last Owner was not deallocated: {before} -> {after}'
gc.collect()
assert ref() is None, 'releasing the C reference did not let the object go'

print('cpyext-lifetime-ok')
"#;

#[test]
fn a_c_reference_roots_its_object_and_releasing_it_lets_it_go() {
    let fixtures = Fixtures::new("cpyext-lifetime");
    fixtures.compile("cpyext_types");
    fixtures.expect_ok(LIFETIME_SCRIPT, &[], "cpyext-lifetime-ok");
}
