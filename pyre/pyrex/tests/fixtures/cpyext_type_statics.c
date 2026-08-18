/* The `PyTypeObject` statics an extension names by address: what each one is
   bound to, and the three things C does with the address. */

#include <Python.h>

#include <string.h>

/* The pending exception's class name and message, taken and cleared. */
static PyObject *pending(void)
{
    PyObject *value = PyErr_GetRaisedException();
    if (value == NULL) {
        Py_RETURN_NONE;
    }
    PyObject *text = PyObject_Str(value);
    PyObject *pair = Py_BuildValue("(sO)", Py_TYPE(value)->tp_name,
                                   text == NULL ? Py_None : text);
    Py_XDECREF(text);
    Py_DECREF(value);
    return pair;
}

/* ── what each static is bound to ─────────────────────────────────────── */

/* `(symbol, tp_name)` for one static; an unbound one has a NULL `tp_name`. */
#define ENTRY(name)                                                          \
    do {                                                                     \
        PyObject *row = Py_BuildValue("(sz)", #name, (&name)->tp_name);      \
        if (row == NULL || PyList_Append(rows, row) < 0) {                   \
            Py_XDECREF(row);                                                 \
            Py_DECREF(rows);                                                 \
            return NULL;                                                     \
        }                                                                    \
        Py_DECREF(row);                                                      \
    } while (0)

static PyObject *statics(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    PyObject *rows = PyList_New(0);
    if (rows == NULL) {
        return NULL;
    }
    ENTRY(PyType_Type);
    ENTRY(PyBaseObject_Type);
    ENTRY(PySuper_Type);
    ENTRY(PyBool_Type);
    ENTRY(PyByteArray_Type);
    ENTRY(PyBytes_Type);
    ENTRY(PyComplex_Type);
    ENTRY(PyDict_Type);
    ENTRY(PyEllipsis_Type);
    ENTRY(PyFloat_Type);
    ENTRY(PyFrozenSet_Type);
    ENTRY(PyList_Type);
    ENTRY(PyLong_Type);
    ENTRY(PyMemoryView_Type);
    ENTRY(PyModule_Type);
    ENTRY(PySet_Type);
    ENTRY(PySlice_Type);
    ENTRY(PyTuple_Type);
    ENTRY(PyUnicode_Type);
    ENTRY(Py_GenericAliasType);
    ENTRY(PyDictProxy_Type);
    ENTRY(PyDictItems_Type);
    ENTRY(PyDictKeys_Type);
    ENTRY(PyDictValues_Type);
    ENTRY(PyClassMethodDescr_Type);
    ENTRY(PyClassMethod_Type);
    ENTRY(PyFunction_Type);
    ENTRY(PyGetSetDescr_Type);
    ENTRY(PyMemberDescr_Type);
    ENTRY(PyMethodDescr_Type);
    ENTRY(PyMethod_Type);
    ENTRY(PyProperty_Type);
    ENTRY(PyStaticMethod_Type);
    ENTRY(PyWrapperDescr_Type);
    ENTRY(PyEnum_Type);
    ENTRY(PyFilter_Type);
    ENTRY(PyMap_Type);
    ENTRY(PyRange_Type);
    ENTRY(PyReversed_Type);
    ENTRY(PyZip_Type);
    ENTRY(PyAsyncGen_Type);
    ENTRY(PyCell_Type);
    ENTRY(PyCode_Type);
    ENTRY(PyCoro_Type);
    ENTRY(PyFrame_Type);
    ENTRY(PyGen_Type);
    ENTRY(PyTraceBack_Type);
    ENTRY(_PyAsyncGenASend_Type);
    ENTRY(_PyWeakref_RefType);
    return rows;
}

#undef ENTRY

/* ── the address as an identity ───────────────────────────────────────── */

/* The static a value's `ob_type` is, named, or `None` for none of them.

   The point of the block being static storage is that it is the one mirror the
   runtime hands out for that type: a synthesized second block would compare
   unequal here while still answering the same `tp_name`. */
static PyObject *type_is(PyObject *self, PyObject *value)
{
    (void)self;
    PyTypeObject *of = Py_TYPE(value);
    struct {
        const char *name;
        PyTypeObject *type;
    } known[] = {
        {"PyType_Type", &PyType_Type},
        {"PyBaseObject_Type", &PyBaseObject_Type},
        {"PyBool_Type", &PyBool_Type},
        {"PyBytes_Type", &PyBytes_Type},
        {"PyDict_Type", &PyDict_Type},
        {"PyEllipsis_Type", &PyEllipsis_Type},
        {"PyFloat_Type", &PyFloat_Type},
        {"PyFunction_Type", &PyFunction_Type},
        {"PyList_Type", &PyList_Type},
        {"PyLong_Type", &PyLong_Type},
        {"PyModule_Type", &PyModule_Type},
        {"PyRange_Type", &PyRange_Type},
        {"PySet_Type", &PySet_Type},
        {"PySlice_Type", &PySlice_Type},
        {"PyTuple_Type", &PyTuple_Type},
        {"PyUnicode_Type", &PyUnicode_Type},
    };
    for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); i++) {
        if (of == known[i].type) {
            return PyUnicode_FromString(known[i].name);
        }
    }
    Py_RETURN_NONE;
}

/* `Py_IS_TYPE`, `PyObject_TypeCheck` and `PyType_IsSubtype` against the same
   static, which is what `PyList_CheckExact` and `PyList_Check` expand to
   where an extension does not call the entry point. */
static PyObject *list_checks(PyObject *self, PyObject *value)
{
    (void)self;
    return Py_BuildValue(
        "(iii)", Py_IS_TYPE(value, &PyList_Type) ? 1 : 0,
        PyObject_TypeCheck(value, &PyList_Type) ? 1 : 0,
        PyType_IsSubtype(Py_TYPE(value), &PyList_Type) ? 1 : 0);
}

/* `PyType_HasFeature` over a static, and the flags it reports. */
static PyObject *type_flags(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    return Py_BuildValue(
        "(iii)", PyType_HasFeature(&PyList_Type, Py_TPFLAGS_READY) ? 1 : 0,
        PyType_HasFeature(&PyList_Type, Py_TPFLAGS_HEAPTYPE) ? 1 : 0,
        PyType_HasFeature(&PyDict_Type, Py_TPFLAGS_BASETYPE) ? 1 : 0);
}

/* ── the address as a converter argument ──────────────────────────────── */

/* `O!`, which takes the static's address and names the type of whatever it
   admitted.  A value of the wrong type is what the error row measures. */
static PyObject *parse_typed(PyObject *self, PyObject *args)
{
    (void)self;
    const char *which;
    PyObject *value;
    if (!PyArg_ParseTuple(args, "sO", &which, &value)) {
        return NULL;
    }
    PyObject *admitted = NULL;
    PyObject *pair = Py_BuildValue("(sO)", which, value);
    if (pair == NULL) {
        return NULL;
    }
    int rc;
    if (strcmp(which, "list") == 0) {
        rc = PyArg_ParseTuple(pair, "sO!", &which, &PyList_Type, &admitted);
    } else if (strcmp(which, "dict") == 0) {
        rc = PyArg_ParseTuple(pair, "sO!", &which, &PyDict_Type, &admitted);
    } else if (strcmp(which, "type") == 0) {
        rc = PyArg_ParseTuple(pair, "sO!", &which, &PyType_Type, &admitted);
    } else {
        Py_DECREF(pair);
        PyErr_SetString(PyExc_ValueError, "the fixture does not offer that type");
        return NULL;
    }
    Py_DECREF(pair);
    if (!rc) {
        return Py_BuildValue("(iOO)", 0, Py_None, pending());
    }
    return Py_BuildValue("(isO)", 1, Py_TYPE(admitted)->tp_name, Py_None);
}

/* ── the address as a base ────────────────────────────────────────────── */

/* A heap type whose base is one of the statics, built the way `PyType_Spec`
   spells it, and asked what it inherited. */
static PyObject *derive_from_dict(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    PyType_Slot slots[] = {
        {Py_tp_base, &PyDict_Type},
        {0, NULL},
    };
    PyType_Spec spec = {
        "cpyext_type_statics.DictSubclass", (int)sizeof(PyObject), 0,
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, slots,
    };
    PyObject *derived = PyType_FromSpec(&spec);
    if (derived == NULL) {
        return pending();
    }
    PyObject *answer = Py_BuildValue(
        "(sii)", ((PyTypeObject *)derived)->tp_name,
        PyType_IsSubtype((PyTypeObject *)derived, &PyDict_Type) ? 1 : 0,
        ((PyTypeObject *)derived)->tp_base == &PyDict_Type ? 1 : 0);
    Py_DECREF(derived);
    return answer;
}

static PyMethodDef methods[] = {
    {"statics", statics, METH_NOARGS, NULL},
    {"type_is", type_is, METH_O, NULL},
    {"list_checks", list_checks, METH_O, NULL},
    {"type_flags", type_flags, METH_NOARGS, NULL},
    {"parse_typed", parse_typed, METH_VARARGS, NULL},
    {"derive_from_dict", derive_from_dict, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_type_statics", NULL, -1,
                                 methods};

PyMODINIT_FUNC PyInit_cpyext_type_statics(void)
{
    return PyModule_Create(&def);
}
