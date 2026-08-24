//! The metaclass a type is built through: a metatype defined by a spec whose
//! only base is `type`, the type built with it, and what the runtime refuses
//! to build a type through.
//!
//! The shape is Cython's, whose every module builds a metatype from a spec and
//! then creates its shared function types through it, so the refusals have to
//! turn away exactly what they name and nothing beside it.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import gc

import cpyext_metaclass as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


# ── the metatype and the type built through it ─────────────────────────

eq('the metatype is a subclass of type', issubclass(m.Meta, type), True)
eq('the metatype mro', m.Meta.__mro__, (m.Meta, type, object))
eq('the metatype name', m.Meta.__name__, 'Meta')
eq('the metatype is itself made by type', type(m.Meta) is type, True)

eq('the type is an instance of the metatype', type(m.T) is m.Meta, True)
eq('isinstance agrees', isinstance(m.T, m.Meta), True)
eq('the type is still a type', isinstance(m.T, type), True)
eq('the type name', m.T.__name__, 'T')
eq('the type mro', m.T.__mro__, (m.T, object))

# The type it built makes instances of its own, which is what it is for.
eq('an instance of the built type', type(m.T(5)) is m.T, True)
eq('the instance value', m.T(5).value, 5)

# ── what the metatype answers for the type ─────────────────────────────

# `tag` is declared by the metatype alone, and it is an attribute of the type
# built through it.
eq('a metatype getset answers for the type', m.T.tag, 'through the metatype')
eq('and the type declares no tag', 'tag' in m.T.__dict__, False)

# `kind` is declared by both.  A getset is a data descriptor, so the metatype's
# is what the type object answers with, while an instance of the type reaches
# the type's own.
eq('the metatype wins for the type', m.T.kind, 'the metatype answers')
eq('the type wins for its instances', m.T(1).kind(), 'the type answers')

# ── the same ob_type read from C ───────────────────────────────────────

# The `ob_type` an extension reads through the mirror is the metatype itself,
# rather than something recorded beside it.
eq('the mirror ob_type is the metatype', m.ob_type_of(m.T, m.Meta), (1, m.Meta))
eq('one built through no metaclass', m.ob_type_of(m.OpenMeta, m.Meta), (0, type))

# ── a metatype that declares no tp_new ─────────────────────────────────

# `PyType_Ready` fills an unset `tp_new` in for the type it readies, so a
# refusal that asks only whether the metatype's `tp_new` differs from `type`'s
# turns away every metatype written in C -- this one, and the one Cython builds
# its shared function types through.  Only a `Py_tp_new` the spec itself names
# is refused, which is what the last of the three refusals below pins.
fresh = m.make_type_through(m.Meta)
eq('a metatype declaring no tp_new is accepted', type(fresh) is m.Meta, True)
eq('the type it built', fresh.__name__, 'Late')

# ── calling the metatype ───────────────────────────────────────────────

# What pins the metatype's instance layout: an instance of a metatype is a
# type, so calling one has to make a class.  A metatype laid out for ordinary
# instances fails here and passes everything else in this file.
made = m.OpenMeta('MadeInPython', (), {})
eq('a class made by calling the metatype', type(made) is m.OpenMeta, True)
eq('the class it made', made.__name__, 'MadeInPython')
eq('it is a class', isinstance(made, type), True)
eq('its ob_type from C', m.ob_type_of(made, m.OpenMeta), (1, m.OpenMeta))
eq('the metatype answers for it too', made.tag, 'through the metatype')
eq('and it makes instances', type(made()) is made, True)

# ── what is refused ────────────────────────────────────────────────────

try:
    m.reject_int_metaclass()
except TypeError as error:
    eq('a metaclass that is not a subclass of type', str(error),
       "Metaclass '<class 'int'>' is not a subclass of 'type'.")
else:
    raise AssertionError('a metaclass that is not a subclass of type was accepted')

try:
    m.reject_metaclass_conflict()
except TypeError as error:
    eq('a metaclass unrelated to the base metatype', str(error),
       'metaclass conflict: the metaclass of a derived class must be a '
       '(non-strict) subclass of the metaclasses of all its bases')
else:
    raise AssertionError('two unrelated metatypes were accepted')

try:
    m.reject_custom_tp_new()
except TypeError as error:
    eq('a metaclass that declares tp_new', str(error),
       'Metaclasses with custom tp_new are not supported.')
else:
    raise AssertionError('a metaclass that declares tp_new was accepted')

# The two the conflict is between: the metatype the call names, and the one the
# base it inherits from carries.
eq('the metatype the call names', type(m.MetaA) is type, True)
eq('the metatype the base carries', type(m.BaseB) is m.MetaB, True)

# ── across a collection ────────────────────────────────────────────────

# The metatype is in no namespace of the module's: the reference handed back
# here is the only one there is, and it has to survive the collection that runs
# before anything is built through it.
late_meta = m.make_metatype()
gc.collect()
late = m.make_type_through(late_meta)
eq('a type built through a collected-over metatype', type(late) is late_meta, True)
eq('its metatype getset answers', late.tag, 'through the metatype')

gc.collect()
eq('the ob_type after another collection', m.ob_type_of(late, late_meta),
   (1, late_meta))
eq('the pair from module exec still holds', type(m.T) is m.Meta, True)
eq('and its metatype getset still answers', m.T.tag, 'through the metatype')

print('cpyext-metaclass-ok')
"#;

#[test]
fn builds_a_type_through_a_c_metatype() {
    let fixtures = Fixtures::new("cpyext-metaclass");
    fixtures.compile("cpyext_metaclass");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-metaclass-ok");
}
