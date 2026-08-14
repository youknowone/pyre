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

print('cpyext-types-ok')
"#;

#[test]
fn defines_types_from_c() {
    let fixtures = Fixtures::new("cpyext-types");
    fixtures.compile("cpyext_types");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-types-ok");
}
