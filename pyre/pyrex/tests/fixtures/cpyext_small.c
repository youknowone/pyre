/* The small entry points an extension reaches for in passing: the unqualified
   type name, the repr recursion guard, the locale codec, a string built from
   code points, the buffer hash, `setdefault` and `origin[args]`. */

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

/* What an entry point answering an object left behind. */
static PyObject *outcome(PyObject *answer)
{
    if (answer == NULL) {
        return pending();
    }
    PyObject *pair = Py_BuildValue("(OO)", answer, Py_None);
    Py_DECREF(answer);
    return pair;
}

/* ── the unqualified type name ────────────────────────────────────────── */

/* `tp_name` beside `_PyType_Name`, which is its tail. */
static PyObject *type_names(PyObject *self, PyObject *value)
{
    (void)self;
    PyTypeObject *type = (PyTypeObject *)value;
    return Py_BuildValue("(ss)", type->tp_name, _PyType_Name(type));
}

/* ── the repr recursion guard ─────────────────────────────────────────── */

/* Enter, ask again, leave, ask once more.  The middle answer is the one that
   says the guard is shared with whatever else is rendering. */
static PyObject *repr_guard(PyObject *self, PyObject *value)
{
    (void)self;
    int first = Py_ReprEnter(value);
    int again = Py_ReprEnter(value);
    if (again == 0) {
        Py_ReprLeave(value);
    }
    Py_ReprLeave(value);
    int after = Py_ReprEnter(value);
    if (after == 0) {
        Py_ReprLeave(value);
    }
    return Py_BuildValue("(iii)", first, again, after);
}

/* A `tp_repr` written the way a recursive container's is: the guard decides
   whether the body is rendered at all. */
static PyObject *guarded_repr(PyObject *self, PyObject *value)
{
    (void)self;
    if (Py_ReprEnter(value)) {
        return PyUnicode_FromString("...");
    }
    PyObject *inner = PyObject_Repr(value);
    Py_ReprLeave(value);
    return inner;
}

/* ── the locale codec ─────────────────────────────────────────────────── */

static PyObject *decode_locale(PyObject *self, PyObject *args)
{
    (void)self;
    const char *data;
    Py_ssize_t length;
    PyObject *errors;
    if (!PyArg_ParseTuple(args, "y#O", &data, &length, &errors)) {
        return NULL;
    }
    const char *handler = errors == Py_None ? NULL : PyUnicode_AsUTF8(errors);
    return outcome(PyUnicode_DecodeLocaleAndSize(data, length, handler));
}

static PyObject *decode_locale_str(PyObject *self, PyObject *args)
{
    (void)self;
    const char *data;
    PyObject *errors;
    if (!PyArg_ParseTuple(args, "yO", &data, &errors)) {
        return NULL;
    }
    const char *handler = errors == Py_None ? NULL : PyUnicode_AsUTF8(errors);
    return outcome(PyUnicode_DecodeLocale(data, handler));
}

static PyObject *encode_locale(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *value;
    PyObject *errors;
    if (!PyArg_ParseTuple(args, "OO", &value, &errors)) {
        return NULL;
    }
    const char *handler = errors == Py_None ? NULL : PyUnicode_AsUTF8(errors);
    return outcome(PyUnicode_EncodeLocale(value, handler));
}

/* ── a string from code points ────────────────────────────────────────── */

/* The caller names the width and hands the points as a sequence of ints, so
   one call covers all three of `PyUnicode_FromKindAndData`'s arms. */
static PyObject *from_kind(PyObject *self, PyObject *args)
{
    (void)self;
    int kind;
    PyObject *points;
    if (!PyArg_ParseTuple(args, "iO", &kind, &points)) {
        return NULL;
    }
    Py_ssize_t size = PySequence_Size(points);
    if (size < 0) {
        return NULL;
    }
    Py_UCS4 wide[64];
    Py_UCS2 narrow[64];
    Py_UCS1 bytes[64];
    if (size > 64) {
        PyErr_SetString(PyExc_ValueError, "the fixture holds at most 64 points");
        return NULL;
    }
    for (Py_ssize_t index = 0; index < size; index++) {
        PyObject *item = PySequence_GetItem(points, index);
        if (item == NULL) {
            return NULL;
        }
        long point = PyLong_AsLong(item);
        Py_DECREF(item);
        if (point == -1 && PyErr_Occurred()) {
            return NULL;
        }
        wide[index] = (Py_UCS4)point;
        narrow[index] = (Py_UCS2)point;
        bytes[index] = (Py_UCS1)point;
    }
    const void *data = kind == PyUnicode_1BYTE_KIND ? (const void *)bytes
                     : kind == PyUnicode_2BYTE_KIND ? (const void *)narrow
                                                    : (const void *)wide;
    return outcome(PyUnicode_FromKindAndData(kind, data, size));
}

/* A negative size and an unknown width, which are the two refusals. */
static PyObject *from_kind_bad(PyObject *self, PyObject *args)
{
    (void)self;
    int kind;
    Py_ssize_t size;
    if (!PyArg_ParseTuple(args, "in", &kind, &size)) {
        return NULL;
    }
    Py_UCS4 one[1] = {65};
    return outcome(PyUnicode_FromKindAndData(kind, one, size));
}

/* ── the buffer hash ──────────────────────────────────────────────────── */

static PyObject *hash_buffer(PyObject *self, PyObject *value)
{
    (void)self;
    char *data;
    Py_ssize_t length;
    if (PyBytes_AsStringAndSize(value, &data, &length) < 0) {
        return NULL;
    }
    return PyLong_FromSsize_t(Py_HashBuffer(data, length));
}

/* ── setdefault ───────────────────────────────────────────────────────── */

static PyObject *set_default(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *mapping;
    PyObject *key;
    PyObject *value;
    if (!PyArg_ParseTuple(args, "OOO", &mapping, &key, &value)) {
        return NULL;
    }
    PyObject *result = NULL;
    int answer = PyDict_SetDefaultRef(mapping, key, value, &result);
    PyObject *row = Py_BuildValue("(iON)", answer,
                                  result == NULL ? Py_None : result, pending());
    Py_XDECREF(result);
    return row;
}

/* The NULL-result spelling, which is the caller wanting only the insertion. */
static PyObject *set_default_no_result(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *mapping;
    PyObject *key;
    PyObject *value;
    if (!PyArg_ParseTuple(args, "OOO", &mapping, &key, &value)) {
        return NULL;
    }
    int answer = PyDict_SetDefaultRef(mapping, key, value, NULL);
    return Py_BuildValue("(iN)", answer, pending());
}

/* ── origin[args] ─────────────────────────────────────────────────────── */

static PyObject *generic_alias(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *origin;
    PyObject *parameters;
    if (!PyArg_ParseTuple(args, "OO", &origin, &parameters)) {
        return NULL;
    }
    return outcome(Py_GenericAlias(origin, parameters));
}

static PyMethodDef methods[] = {
    {"type_names", type_names, METH_O, NULL},
    {"repr_guard", repr_guard, METH_O, NULL},
    {"guarded_repr", guarded_repr, METH_O, NULL},
    {"decode_locale", decode_locale, METH_VARARGS, NULL},
    {"decode_locale_str", decode_locale_str, METH_VARARGS, NULL},
    {"encode_locale", encode_locale, METH_VARARGS, NULL},
    {"from_kind", from_kind, METH_VARARGS, NULL},
    {"from_kind_bad", from_kind_bad, METH_VARARGS, NULL},
    {"hash_buffer", hash_buffer, METH_O, NULL},
    {"set_default", set_default, METH_VARARGS, NULL},
    {"set_default_no_result", set_default_no_result, METH_VARARGS, NULL},
    {"generic_alias", generic_alias, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_small", NULL, -1,
                                 methods};

PyMODINIT_FUNC PyInit_cpyext_small(void)
{
    return PyModule_Create(&def);
}
