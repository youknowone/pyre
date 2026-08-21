//! The namespace an extension reaches through `tp_dict`: that the field is the
//! type's own, that a write through it answers from Python, and that it
//! survives the collection that moves the dict it names.
//!
//! The shape is `__Pyx_setup_reduce`'s, which every Cython module runs against
//! every extension type it defines.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import gc

import cpyext_type_dict as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


for kind in (m.Subject, m.Static):
    label = kind.__name__

    # ── the field is the namespace, not a copy of it ───────────────────

    # `type.__dict__` reports a proxy over the same mapping, so the two agree
    # key for key rather than by identity.
    eq('%s keys' % label, sorted(m.type_dict(kind)), sorted(kind.__dict__))
    eq('%s declared' % label, m.declares(kind, 'declared'), 1)
    eq('%s undeclared' % label, m.declares(kind, 'nowhere'), 0)

    # ── a write through it answers from Python ─────────────────────────

    m.set_on_type_dict(kind, 'label', 'written through tp_dict')
    eq('%s reads back' % label, kind.label, 'written through tp_dict')
    eq('%s instances read it' % label, kind().label, 'written through tp_dict')
    eq('%s __dict__ has it' % label, kind.__dict__['label'],
       'written through tp_dict')

    # And a delete takes it away again.
    m.del_from_type_dict(kind, 'label')
    eq('%s gone from the namespace' % label, 'label' in kind.__dict__, False)
    try:
        kind.label
    except AttributeError:
        pass
    else:
        raise AssertionError('%s: a deleted key still answered' % label)

    # ── the rename Cython performs ─────────────────────────────────────

    eq('%s starts with the cython name' % label,
       m.declares(kind, '__reduce_cython__'), 1)
    m.rename_on_type_dict(kind, '__reduce_cython__', '__reduce__')
    eq('%s renamed' % label, m.declares(kind, '__reduce__'), 1)
    eq('%s old name gone' % label, m.declares(kind, '__reduce_cython__'), 0)
    eq('%s renamed answers' % label, kind().__reduce__(),
       'declared on the type')

    # ── across a collection ────────────────────────────────────────────

    # The dict the field names moves; the block the field holds does not.
    gc.collect()
    eq('%s keys after a collection' % label,
       sorted(m.type_dict(kind)), sorted(kind.__dict__))
    m.set_on_type_dict(kind, 'late', 'after a collection')
    eq('%s written after a collection' % label, kind.late,
       'after a collection')
    gc.collect()
    eq('%s still written' % label, kind.late, 'after a collection')
    eq('%s still declares it' % label, m.declares(kind, 'late'), 1)

# A subclass reaches what was written on its base, and writes on the base
# after the subclass exists still reach it.
class Derived(m.Subject):
    pass


m.set_on_type_dict(m.Subject, 'shared', 'from the base')
eq('subclass reads the base', Derived.shared, 'from the base')
eq('subclass instance reads the base', Derived().shared, 'from the base')
m.del_from_type_dict(m.Subject, 'shared')
eq('subclass loses it too', hasattr(Derived, 'shared'), False)

print('cpyext-type-dict-ok')
"#;

#[test]
fn the_namespace_an_extension_writes_through() {
    let fixtures = Fixtures::new("cpyext-type-dict");
    fixtures.compile("cpyext_type_dict");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-type-dict-ok");
}
