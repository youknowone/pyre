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
assert m.Point.__doc__ == 'a two-dimensional point defined in C'
assert isinstance(m.Point, type)
assert m.flags() == (1, 1)

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
assert repr(m.Point.__dict__['x']) == "<attribute 'x' of 'Point' objects>"
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
assert repr(m.Point.__dict__['total']) == "<attribute 'total' of 'Point' objects>"
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

print('cpyext-types-ok')
"#;

#[test]
fn defines_types_from_c() {
    let fixtures = Fixtures::new("cpyext-types");
    fixtures.compile("cpyext_types");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-types-ok");
}
