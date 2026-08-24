/* The metaclass an extension builds its types through: a metatype defined by
   a spec whose only base is `type`, and the types created with it -- the
   sequence `__Pyx_FetchCommonTypeFromSpec` runs for the function types every
   Cython module shares. */
#include "Python.h"
#include <string.h>

/* ── Meta: the metatype, in the shape Cython declares one ───────────── */

static PyObject *meta_get_module(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyUnicode_FromString("cpyext_metaclass_abi");
}

static PyObject *meta_get_tag(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyUnicode_FromString("through the metatype");
}

static PyObject *meta_get_kind(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyUnicode_FromString("the metatype answers");
}

/* The table the metatype carries.  `__module__` is the name Cython's declares;
   `tag` is named here and nowhere else, and `kind` is named again by the type
   below. */
static PyGetSetDef meta_getset[] = {
    {"__module__", meta_get_module, NULL, NULL, NULL},
    {"tag", meta_get_tag, NULL, NULL, NULL},
    {"kind", meta_get_kind, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

/* No `Py_tp_new`: a metatype is what a type is built through, and one that
   names its own is refused. */
static PyType_Slot meta_slots[] = {
    {Py_tp_getset, (void *)meta_getset},
    {0, NULL},
};

static PyType_Spec meta_spec = {
    "cpyext_metaclass.Meta",
    0,
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DISALLOW_INSTANTIATION,
    meta_slots,
};

/* The same metatype without `Py_TPFLAGS_DISALLOW_INSTANTIATION`, which Python
   can call to make a class of its own. */
static PyType_Spec open_meta_spec = {
    "cpyext_metaclass.OpenMeta",
    0,
    0,
    Py_TPFLAGS_DEFAULT,
    meta_slots,
};

/* Built after the module is running rather than while it is being created, so
   a collection can happen between the metatype and the type. */
static PyType_Spec late_meta_spec = {
    "cpyext_metaclass.LateMeta",
    0,
    0,
    Py_TPFLAGS_DEFAULT,
    meta_slots,
};

/* ── T: a type built through the metatype ───────────────────────────── */

typedef struct {
    PyObject_HEAD
    long value;
} ThroughObject;

static PyObject *through_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *keywords[] = {"value", NULL};
    ThroughObject *self;
    long value = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|l", keywords, &value)) {
        return NULL;
    }
    self = (ThroughObject *)PyType_GenericAlloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    self->value = value;
    return (PyObject *)self;
}

/* `kind` again, on the type this time: the metatype's getset is a data
   descriptor, so the type object answers with that one, while an instance of
   the type reaches this one. */
static PyObject *through_kind(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    return PyUnicode_FromString("the type answers");
}

static PyMethodDef through_methods[] = {
    {"kind", through_kind, METH_NOARGS, "which namespace answered"},
    {NULL, NULL, 0, NULL},
};

static PyMemberDef through_members[] = {
    {"value", Py_T_LONG, offsetof(ThroughObject, value), 0, "the value"},
    {NULL, 0, 0, 0, NULL},
};

static PyType_Slot through_slots[] = {
    {Py_tp_new, (void *)through_new},
    {Py_tp_methods, (void *)through_methods},
    {Py_tp_members, (void *)through_members},
    {0, NULL},
};

static PyType_Spec through_spec = {
    "cpyext_metaclass.T",
    sizeof(ThroughObject),
    0,
    Py_TPFLAGS_DEFAULT,
    through_slots,
};

/* ── the metatypes the refusals are about ───────────────────────────── */

static PyObject *meta_with_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    (void)type;
    (void)args;
    (void)kwds;
    PyErr_SetString(PyExc_TypeError, "this metatype makes nothing");
    return NULL;
}

static PyType_Slot meta_with_new_slots[] = {
    {Py_tp_new, (void *)meta_with_new},
    {0, NULL},
};

static PyType_Spec meta_with_new_spec = {
    "cpyext_metaclass.MetaWithNew",
    0,
    0,
    Py_TPFLAGS_DEFAULT,
    meta_with_new_slots,
};

static PyType_Slot bare_slots[] = {
    {0, NULL},
};

/* Two metatypes with nothing between them, and a type that is an instance of
   the second one. */
static PyType_Spec meta_a_spec = {
    "cpyext_metaclass.MetaA",
    0,
    0,
    Py_TPFLAGS_DEFAULT,
    bare_slots,
};

static PyType_Spec meta_b_spec = {
    "cpyext_metaclass.MetaB",
    0,
    0,
    Py_TPFLAGS_DEFAULT,
    bare_slots,
};

static PyType_Spec base_b_spec = {
    "cpyext_metaclass.BaseB",
    0,
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    bare_slots,
};

/* The spec each refused call is written against, which none of them builds. */
static PyType_Spec refused_spec = {
    "cpyext_metaclass.Refused",
    0,
    0,
    Py_TPFLAGS_DEFAULT,
    bare_slots,
};

static PyType_Spec late_spec = {
    "cpyext_metaclass.Late",
    0,
    0,
    Py_TPFLAGS_DEFAULT,
    bare_slots,
};

/* The types the entry points below name.  Each is a reference this module
   keeps for as long as it is loaded; the module holds one of its own. */
static PyObject *meta_type;
static PyObject *meta_with_new_type;
static PyObject *meta_a_type;
static PyObject *base_b_type;

/* ── what the module exports ────────────────────────────────────────── */

/* Whether the `ob_type` the extension reads through the mirror is `metatype`,
   beside the type it actually holds. */
static PyObject *ob_type_of(PyObject *self, PyObject *args)
{
    PyObject *subject;
    PyObject *metatype;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &subject, &metatype)) {
        return NULL;
    }
    return Py_BuildValue("iO", Py_TYPE(subject) == (PyTypeObject *)metatype,
                         (PyObject *)Py_TYPE(subject));
}

/* A metatype handed back before anything is built through it. */
static PyObject *make_metatype(PyObject *self, PyObject *unused)
{
    (void)unused;
    return PyType_FromMetaclass(NULL, self, &late_meta_spec,
                                (PyObject *)&PyType_Type);
}

/* A type built through the metatype the caller hands over. */
static PyObject *make_type_through(PyObject *self, PyObject *metatype)
{
    if (!PyType_Check(metatype)) {
        PyErr_SetString(PyExc_TypeError, "not a type");
        return NULL;
    }
    return PyType_FromMetaclass((PyTypeObject *)metatype, self, &late_spec, NULL);
}

/* Report a call that had to fail: the error it left is the answer, and a type
   it built instead is this fixture's own failure to report. */
static PyObject *refused(PyObject *made, const char *complaint)
{
    if (made != NULL) {
        Py_DECREF(made);
        PyErr_SetString(PyExc_AssertionError, complaint);
    }
    return NULL;
}

/* `int` is a type, but not one whose instances are types. */
static PyObject *reject_int_metaclass(PyObject *self, PyObject *unused)
{
    (void)unused;
    return refused(PyType_FromMetaclass(&PyLong_Type, self, &refused_spec, NULL),
                   "a metaclass that is not a subclass of type was accepted");
}

/* `MetaA` is named and `BaseB` is inherited from, and `BaseB`'s own metatype
   is `MetaB`: neither of the two derives from the other. */
static PyObject *reject_metaclass_conflict(PyObject *self, PyObject *unused)
{
    (void)unused;
    return refused(PyType_FromMetaclass((PyTypeObject *)meta_a_type, self,
                                        &refused_spec, base_b_type),
                   "two unrelated metatypes were accepted");
}

/* The type is built here, so a `__new__` the metatype defines never runs. */
static PyObject *reject_custom_tp_new(PyObject *self, PyObject *unused)
{
    (void)unused;
    return refused(PyType_FromMetaclass((PyTypeObject *)meta_with_new_type, self,
                                        &refused_spec, NULL),
                   "a metaclass that declares tp_new was accepted");
}

static PyMethodDef methods[] = {
    {"ob_type_of", ob_type_of, METH_VARARGS, NULL},
    {"make_metatype", make_metatype, METH_NOARGS, NULL},
    {"make_type_through", make_type_through, METH_O, NULL},
    {"reject_int_metaclass", reject_int_metaclass, METH_NOARGS, NULL},
    {"reject_metaclass_conflict", reject_metaclass_conflict, METH_NOARGS, NULL},
    {"reject_custom_tp_new", reject_custom_tp_new, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

/* Build a type through `metaclass` and publish it under the last component of
   its name.  The reference returned is the caller's; the module holds one of
   its own. */
static PyObject *add_type(PyObject *module, PyTypeObject *metaclass,
                          PyType_Spec *spec, PyObject *bases)
{
    PyObject *made;
    const char *name;
    made = PyType_FromMetaclass(metaclass, module, spec, bases);
    if (made == NULL) {
        return NULL;
    }
    name = strrchr(spec->name, '.');
    name = name ? name + 1 : spec->name;
    if (PyModule_AddObjectRef(module, name, made) < 0) {
        Py_DECREF(made);
        return NULL;
    }
    return made;
}

static int metaclass_exec(PyObject *module)
{
    PyObject *bases;
    PyObject *made;

    /* `__pyx_CommonTypesMetaclass_init` verbatim: no metaclass of its own, the
       module it belongs to, and `type` packed into a tuple as its only base. */
    bases = PyTuple_Pack(1, (PyObject *)&PyType_Type);
    if (bases == NULL) {
        return -1;
    }
    meta_type = add_type(module, NULL, &meta_spec, bases);
    Py_DECREF(bases);
    if (meta_type == NULL) {
        return -1;
    }
    /* And the type the metatype exists for, which is what the module exports
       -- `__Pyx_FetchCommonTypeFromSpec` naming the metatype it just made. */
    made = add_type(module, (PyTypeObject *)meta_type, &through_spec, NULL);
    if (made == NULL) {
        return -1;
    }
    Py_DECREF(made);
    /* The bases argument the other way round: a single type rather than a
       tuple, which is the other spelling the entry point takes. */
    made = add_type(module, NULL, &open_meta_spec, (PyObject *)&PyType_Type);
    if (made == NULL) {
        return -1;
    }
    Py_DECREF(made);
    meta_with_new_type = add_type(module, NULL, &meta_with_new_spec,
                                  (PyObject *)&PyType_Type);
    if (meta_with_new_type == NULL) {
        return -1;
    }
    meta_a_type = add_type(module, NULL, &meta_a_spec, (PyObject *)&PyType_Type);
    if (meta_a_type == NULL) {
        return -1;
    }
    made = add_type(module, NULL, &meta_b_spec, (PyObject *)&PyType_Type);
    if (made == NULL) {
        return -1;
    }
    base_b_type = add_type(module, (PyTypeObject *)made, &base_b_spec, NULL);
    Py_DECREF(made);
    if (base_b_type == NULL) {
        return -1;
    }
    return 0;
}

static PyModuleDef_Slot slots[] = {
    {Py_mod_exec, (void *)metaclass_exec},
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpyext_metaclass",
    "pyre cpyext metaclass module",
    0,
    methods,
    slots,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit_cpyext_metaclass(void)
{
    return PyModuleDef_Init(&moduledef);
}
