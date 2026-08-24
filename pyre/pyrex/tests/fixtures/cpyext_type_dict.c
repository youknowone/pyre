/* The namespace an extension reads and writes through `tp_dict`: the sequence
   `__Pyx_setup_reduce` runs on every Cython extension type, which renames
   `__reduce_cython__` to `__reduce__` by writing one key and deleting the
   other. */
#include "Python.h"

/* ── a type built from a spec, and one readied as a static ──────────── */

static PyObject *declared(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    return PyUnicode_FromString("declared on the type");
}

static PyMethodDef subject_methods[] = {
    {"declared", declared, METH_NOARGS, NULL},
    {"__reduce_cython__", declared, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static PyType_Slot subject_slots[] = {
    {Py_tp_methods, (void *)subject_methods},
    {0, NULL},
};

static PyType_Spec subject_spec = {
    "cpyext_type_dict.Subject",
    sizeof(PyObject),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    subject_slots,
};

static PyTypeObject StaticType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "cpyext_type_dict.Static",
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = subject_methods,
    /* A static type that declares no constructor cannot be instantiated
       (`type_ready_set_new`), and the script below builds one. */
    .tp_new = PyType_GenericNew,
};

/* ── what the field answers with ────────────────────────────────────── */

/* The type's own namespace, handed back so Python can compare it against what
   `type.__dict__` reports. */
/* Every entry point below reads `tp_dict` straight out of the block, so the
   argument has to be a type before the read rather than after it. */
static PyTypeObject *as_type(PyObject *object)
{
    if (PyType_Check(object)) {
        return (PyTypeObject *)object;
    }
    PyErr_SetString(PyExc_TypeError, "not a type");
    return NULL;
}

static PyObject *type_dict(PyObject *self, PyObject *object)
{
    PyTypeObject *type = as_type(object);
    PyObject *dict;
    (void)self;
    if (type == NULL) {
        return NULL;
    }
    dict = type->tp_dict;
    if (dict == NULL) {
        PyErr_SetString(PyExc_AssertionError, "tp_dict is NULL");
        return NULL;
    }
    return Py_NewRef(dict);
}

/* Whether the field is a dict carrying `name`, without going through the type
   -- the read `__Pyx_setup_reduce` makes before it writes. */
static PyObject *declares(PyObject *self, PyObject *args)
{
    PyObject *object;
    PyTypeObject *type;
    const char *name;
    PyObject *found;
    (void)self;
    if (!PyArg_ParseTuple(args, "Os", &object, &name)) {
        return NULL;
    }
    type = as_type(object);
    if (type == NULL) {
        return NULL;
    }
    if (!PyDict_Check(type->tp_dict)) {
        PyErr_SetString(PyExc_AssertionError, "tp_dict is not a dict");
        return NULL;
    }
    found = PyDict_GetItemString(type->tp_dict, name);
    if (found == NULL && PyErr_Occurred()) {
        return NULL;
    }
    return PyLong_FromLong(found != NULL);
}

/* `__Pyx__SetItemOnTypeDict` verbatim: the write, then the report that the
   namespace changed. */
static PyObject *set_on_type_dict(PyObject *self, PyObject *args)
{
    PyObject *object;
    PyTypeObject *type;
    PyObject *key;
    PyObject *value;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOO", &object, &key, &value)) {
        return NULL;
    }
    type = as_type(object);
    if (type == NULL) {
        return NULL;
    }
    if (PyDict_SetItem(type->tp_dict, key, value) < 0) {
        return NULL;
    }
    PyType_Modified(type);
    Py_RETURN_NONE;
}

/* `__Pyx__DelItemOnTypeDict` verbatim. */
static PyObject *del_from_type_dict(PyObject *self, PyObject *args)
{
    PyObject *object;
    PyTypeObject *type;
    PyObject *key;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &object, &key)) {
        return NULL;
    }
    type = as_type(object);
    if (type == NULL) {
        return NULL;
    }
    if (PyDict_DelItem(type->tp_dict, key) < 0) {
        return NULL;
    }
    PyType_Modified(type);
    Py_RETURN_NONE;
}

/* The rename itself, as one call: what every Cython module does to each of its
   extension types before the module finishes loading. */
static PyObject *rename_on_type_dict(PyObject *self, PyObject *args)
{
    PyObject *object;
    PyTypeObject *type;
    PyObject *from;
    PyObject *to;
    PyObject *value;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOO", &object, &from, &to)) {
        return NULL;
    }
    type = as_type(object);
    if (type == NULL) {
        return NULL;
    }
    value = PyObject_GetAttr(object, from);
    if (value == NULL) {
        return NULL;
    }
    if (PyDict_SetItem(type->tp_dict, to, value) < 0) {
        Py_DECREF(value);
        return NULL;
    }
    Py_DECREF(value);
    if (PyDict_DelItem(type->tp_dict, from) < 0) {
        return NULL;
    }
    PyType_Modified(type);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"type_dict", type_dict, METH_O, NULL},
    {"declares", declares, METH_VARARGS, NULL},
    {"set_on_type_dict", set_on_type_dict, METH_VARARGS, NULL},
    {"del_from_type_dict", del_from_type_dict, METH_VARARGS, NULL},
    {"rename_on_type_dict", rename_on_type_dict, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static int type_dict_exec(PyObject *module)
{
    PyObject *made;
    made = PyType_FromModuleAndSpec(module, &subject_spec, NULL);
    if (made == NULL) {
        return -1;
    }
    if (PyModule_AddObjectRef(module, "Subject", made) < 0) {
        Py_DECREF(made);
        return -1;
    }
    Py_DECREF(made);
    if (PyType_Ready(&StaticType) < 0) {
        return -1;
    }
    return PyModule_AddObjectRef(module, "Static", (PyObject *)&StaticType);
}

static PyModuleDef_Slot slots[] = {
    {Py_mod_exec, (void *)type_dict_exec},
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpyext_type_dict",
    "pyre cpyext type namespace module",
    0,
    methods,
    slots,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit_cpyext_type_dict(void)
{
    return PyModuleDef_Init(&moduledef);
}
