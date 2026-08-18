//! The `PyTypeObject` statics an extension names by address: what each one is
//! bound to, and the three things C does with the address.
//!
//! Every expectation was taken from CPython 3.14.6 running this same script
//! against this same fixture.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import cpyext_type_statics as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


def fn():
    pass


class L(list):
    pass


class Liar:
    # `__class__` answering with `list` is what separates an `isinstance` test
    # from the layout test `O!` is.
    @property
    def __class__(self):
        return list


# ── what each static is bound to ───────────────────────────────────────

# A NULL `tp_name` here means the static was never bound, which is what an
# extension resolving the symbol and reading through it would find.
eq('statics', m.statics(), [
    ('PyType_Type', 'type'),
    ('PyBaseObject_Type', 'object'),
    ('PySuper_Type', 'super'),
    ('PyBool_Type', 'bool'),
    ('PyByteArray_Type', 'bytearray'),
    ('PyBytes_Type', 'bytes'),
    ('PyComplex_Type', 'complex'),
    ('PyDict_Type', 'dict'),
    ('PyEllipsis_Type', 'ellipsis'),
    ('PyFloat_Type', 'float'),
    ('PyFrozenSet_Type', 'frozenset'),
    ('PyList_Type', 'list'),
    ('PyLong_Type', 'int'),
    ('PyMemoryView_Type', 'memoryview'),
    ('PyModule_Type', 'module'),
    ('PySet_Type', 'set'),
    ('PySlice_Type', 'slice'),
    ('PyTuple_Type', 'tuple'),
    ('PyUnicode_Type', 'str'),
    ('Py_GenericAliasType', 'types.GenericAlias'),
    ('PyDictProxy_Type', 'mappingproxy'),
    ('PyDictItems_Type', 'dict_items'),
    ('PyDictKeys_Type', 'dict_keys'),
    ('PyDictValues_Type', 'dict_values'),
    ('PyClassMethodDescr_Type', 'classmethod_descriptor'),
    ('PyClassMethod_Type', 'classmethod'),
    ('PyFunction_Type', 'function'),
    ('PyGetSetDescr_Type', 'getset_descriptor'),
    ('PyMemberDescr_Type', 'member_descriptor'),
    ('PyMethodDescr_Type', 'method_descriptor'),
    ('PyMethod_Type', 'method'),
    ('PyProperty_Type', 'property'),
    ('PyStaticMethod_Type', 'staticmethod'),
    ('PyWrapperDescr_Type', 'wrapper_descriptor'),
    ('PyEnum_Type', 'enumerate'),
    ('PyFilter_Type', 'filter'),
    ('PyMap_Type', 'map'),
    ('PyRange_Type', 'range'),
    ('PyReversed_Type', 'reversed'),
    ('PyZip_Type', 'zip'),
    ('PyAsyncGen_Type', 'async_generator'),
    ('PyCell_Type', 'cell'),
    ('PyCode_Type', 'code'),
    ('PyCoro_Type', 'coroutine'),
    ('PyFrame_Type', 'frame'),
    ('PyGen_Type', 'generator'),
    ('PyTraceBack_Type', 'traceback'),
    ('_PyAsyncGenASend_Type', 'async_generator_asend'),
    ('_PyWeakref_RefType', 'weakref.ReferenceType'),
])

# ── the address is the identity ────────────────────────────────────────

# The point of static storage: the block C names is the one mirror the runtime
# hands out, so `Py_TYPE(x) == &PyList_Type` holds.  A second, synthesized
# block would answer the same `tp_name` and still fail this.
eq('type of type', m.type_is(type), 'PyType_Type')
eq('type of object', m.type_is(object), 'PyType_Type')
eq('type of bool', m.type_is(True), 'PyBool_Type')
eq('type of bytes', m.type_is(b'x'), 'PyBytes_Type')
eq('type of dict', m.type_is({}), 'PyDict_Type')
eq('type of float', m.type_is(1.0), 'PyFloat_Type')
eq('type of list', m.type_is([]), 'PyList_Type')
eq('type of int', m.type_is(7), 'PyLong_Type')
eq('type of set', m.type_is({1}), 'PySet_Type')
eq('type of slice', m.type_is(slice(1)), 'PySlice_Type')
eq('type of tuple', m.type_is(()), 'PyTuple_Type')
eq('type of str', m.type_is('x'), 'PyUnicode_Type')
eq('type of range', m.type_is(range(3)), 'PyRange_Type')
eq('type of module', m.type_is(m), 'PyModule_Type')
eq('type of function', m.type_is(fn), 'PyFunction_Type')
# The singletons resolve their own `ob_type` when they are bound, so their
# types are the ones a binding order that ran too late would miss.
eq('type of Ellipsis', m.type_is(Ellipsis), 'PyEllipsis_Type')
# A type with no static of its own is none of them.
eq('type of None', m.type_is(None), None)
eq('type of a subclass', m.type_is(L()), None)

# `Py_IS_TYPE` / `PyObject_TypeCheck` / `PyType_IsSubtype` over one static,
# which is what `PyList_CheckExact` and `PyList_Check` expand to.
eq('checks list', m.list_checks([]), (1, 1, 1))
eq('checks subclass', m.list_checks(L()), (0, 1, 1))
eq('checks tuple', m.list_checks(()), (0, 0, 0))
eq('checks str', m.list_checks('x'), (0, 0, 0))

# `PyType_HasFeature` over a static: ready, not a heap type, and a base.
eq('flags', m.type_flags(), (1, 0, 1))

# ── the address as a converter argument ────────────────────────────────

eq('O! list', m.parse_typed('list', [1, 2]), (1, 'list', None))
eq('O! subclass', m.parse_typed('list', L()), (1, 'L', None))
eq('O! dict', m.parse_typed('dict', {'a': 1}), (1, 'dict', None))
eq('O! type', m.parse_typed('type', int), (1, 'type', None))

# The refusals.  `O!` is a layout test, so an object whose `__class__` answers
# with `list` is refused like any other wrong type.  Only the class and the
# tail of the message are compared: pyre's argument parser names the function
# where CPython numbers the argument, which is the parser's own divergence and
# is the same for every format unit.
for name, which, argument, wanted in [
        ('tuple', 'list', (1, 2), 'must be list, not tuple'),
        ('liar', 'list', Liar(), 'must be list, not Liar'),
        ('none', 'list', None, 'must be list, not None'),
        ('list', 'dict', [], 'must be dict, not list'),
        ('int', 'type', 3, 'must be type, not int')]:
    answer = m.parse_typed(which, argument)
    eq('O! refuses %s' % name,
       (answer[0], answer[1], answer[2][0], answer[2][1].endswith(wanted)),
       (0, None, 'TypeError', True))

# ── the address as a base ──────────────────────────────────────────────

# `Py_tp_base` naming a static: the derived type is a `dict` subclass and its
# `tp_base` is the static itself.
eq('derived', m.derive_from_dict(), ('cpyext_type_statics.DictSubclass', 1, 1))

print('cpyext-type-statics-ok')
"#;

#[test]
fn the_builtin_type_statics() {
    let fixtures = Fixtures::new("cpyext-type-statics");
    fixtures.compile("cpyext_type_statics");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-type-statics-ok");
}
