/* A multi-phase (PEP 489) extension exercising the canonical `str`
   representation: the kind/data pair, the read and write macros, and the
   allocate-then-fill shape `PyUnicode_New` exists for. */

#include <Python.h>

static struct PyModuleDef moduledef;

/* The exception an entry point left behind, as `(type name, message)`. */
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

/* The shape a real extension uses: measure the input, allocate a string wide
   enough for it, then fill it through the caller's own pointer. */
static PyObject *u_escape(PyObject *self, PyObject *arg)
{
    (void)self;
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "escape() wants a str");
        return NULL;
    }
    if (PyUnicode_READY(arg) < 0) {
        return NULL;
    }
    Py_ssize_t length = PyUnicode_GET_LENGTH(arg);
    int in_kind = PyUnicode_KIND(arg);
    const void *in_data = PyUnicode_DATA(arg);
    Py_UCS4 maxchar = 0;
    Py_ssize_t out_length = 0;
    for (Py_ssize_t index = 0; index < length; index++) {
        Py_UCS4 point = PyUnicode_READ(in_kind, in_data, index);
        if (point > maxchar) {
            maxchar = point;
        }
        out_length += point == '<' ? 4 : 1;
    }
    if (out_length == length) {
        /* Nothing to escape: hand back the argument itself. */
        Py_INCREF(arg);
        return arg;
    }
    PyObject *out = PyUnicode_New(out_length, maxchar);
    if (out == NULL) {
        return NULL;
    }
    int out_kind = PyUnicode_KIND(out);
    void *out_data = PyUnicode_DATA(out);
    Py_ssize_t at = 0;
    for (Py_ssize_t index = 0; index < length; index++) {
        Py_UCS4 point = PyUnicode_READ(in_kind, in_data, index);
        if (point == '<') {
            PyUnicode_WRITE(out_kind, out_data, at++, '&');
            PyUnicode_WRITE(out_kind, out_data, at++, 'l');
            PyUnicode_WRITE(out_kind, out_data, at++, 't');
            PyUnicode_WRITE(out_kind, out_data, at++, ';');
        }
        else {
            PyUnicode_WRITE(out_kind, out_data, at++, point);
        }
    }
    return out;
}

/* `(kind, is_ascii, max_char_value, length)` for a string. */
static PyObject *u_shape(PyObject *self, PyObject *arg)
{
    (void)self;
    return Py_BuildValue("(iIIn)", PyUnicode_KIND(arg), PyUnicode_IS_ASCII(arg),
                         PyUnicode_MAX_CHAR_VALUE(arg), PyUnicode_GET_LENGTH(arg));
}

/* The typed `*_DATA` casts, chosen by the kind the string reports. */
static PyObject *u_first_point(PyObject *self, PyObject *arg)
{
    (void)self;
    if (PyUnicode_GET_LENGTH(arg) < 1) {
        PyErr_SetString(PyExc_ValueError, "first_point() wants a non-empty str");
        return NULL;
    }
    Py_UCS4 point;
    switch (PyUnicode_KIND(arg)) {
    case PyUnicode_1BYTE_KIND:
        point = PyUnicode_1BYTE_DATA(arg)[0];
        break;
    case PyUnicode_2BYTE_KIND:
        point = PyUnicode_2BYTE_DATA(arg)[0];
        break;
    default:
        point = PyUnicode_4BYTE_DATA(arg)[0];
        break;
    }
    if (point != PyUnicode_READ_CHAR(arg, 0)) {
        PyErr_SetString(PyExc_SystemError, "typed data disagrees with READ_CHAR");
        return NULL;
    }
    return PyLong_FromUnsignedLong(point);
}

/* The entry-point spellings of the same read and write. */
static PyObject *u_reverse(PyObject *self, PyObject *arg)
{
    (void)self;
    Py_ssize_t length = PyUnicode_GetLength(arg);
    if (length < 0) {
        return NULL;
    }
    PyObject *out = PyUnicode_New(length, 0x10ffff);
    if (out == NULL) {
        return NULL;
    }
    for (Py_ssize_t index = 0; index < length; index++) {
        Py_UCS4 point = PyUnicode_ReadChar(arg, length - 1 - index);
        if (point == (Py_UCS4)-1 && PyErr_Occurred()) {
            Py_DECREF(out);
            return NULL;
        }
        if (PyUnicode_WriteChar(out, index, point) < 0) {
            Py_DECREF(out);
            return NULL;
        }
    }
    return out;
}

/* An out-of-range index on either accessor raises IndexError. */
static PyObject *u_out_of_range(PyObject *self, PyObject *arg)
{
    (void)self;
    (void)arg;
    PyObject *text = PyUnicode_FromString("ab");
    if (text == NULL) {
        return NULL;
    }
    if (PyUnicode_ReadChar(text, 2) != (Py_UCS4)-1 || !PyErr_Occurred()) {
        Py_DECREF(text);
        PyErr_SetString(PyExc_SystemError, "ReadChar accepted an out-of-range index");
        return NULL;
    }
    PyErr_Clear();
    if (PyUnicode_WriteChar(text, -1, 'x') != -1 || !PyErr_Occurred()) {
        Py_DECREF(text);
        PyErr_SetString(PyExc_SystemError, "WriteChar accepted a negative index");
        return NULL;
    }
    PyErr_Clear();
    Py_DECREF(text);
    Py_RETURN_TRUE;
}

/* `PyUnicode_New` rejects what it cannot represent. */
static PyObject *u_rejects(PyObject *self, PyObject *arg)
{
    (void)self;
    (void)arg;
    if (PyUnicode_New(-1, 0x7f) != NULL || !PyErr_Occurred()) {
        PyErr_SetString(PyExc_SystemError, "PyUnicode_New accepted a negative size");
        return NULL;
    }
    PyErr_Clear();
    if (PyUnicode_New(1, 0x110000) != NULL || !PyErr_Occurred()) {
        PyErr_SetString(PyExc_SystemError, "PyUnicode_New accepted maxchar past U+10FFFF");
        return NULL;
    }
    PyErr_Clear();
    Py_RETURN_TRUE;
}

/* The width is fixed when the string is made, so a write of a code point
   wider than it holds is refused rather than cut down.  Both halves happen
   here: a string is modifiable only while nothing else names it. */
static PyObject *u_new_and_write(PyObject *self, PyObject *args)
{
    (void)self;
    Py_ssize_t length;
    unsigned long maxchar;
    unsigned long value;
    if (!PyArg_ParseTuple(args, "nkk", &length, &maxchar, &value)) {
        return NULL;
    }
    PyObject *text = PyUnicode_New(length, (Py_UCS4)maxchar);
    if (text == NULL) {
        return NULL;
    }
    for (Py_ssize_t index = 0; index < length; index++) {
        Py_UCS4 point = index == 0 ? (Py_UCS4)value : (Py_UCS4)'x';
        if (PyUnicode_WriteChar(text, index, point) < 0) {
            Py_DECREF(text);
            return NULL;
        }
    }
    return text;
}

/* Neither accessor takes anything but a string, and saying so is what
   `PyErr_BadArgument` is for. */
static PyObject *u_not_a_string(PyObject *self, PyObject *arg)
{
    (void)self;
    if (PyUnicode_ReadChar(arg, 0) != (Py_UCS4)-1 || !PyErr_Occurred()) {
        PyErr_SetString(PyExc_SystemError, "ReadChar accepted something that is not a string");
        return NULL;
    }
    PyObject *read = pending();
    if (PyUnicode_WriteChar(arg, 0, 'x') != -1 || !PyErr_Occurred()) {
        Py_XDECREF(read);
        PyErr_SetString(PyExc_SystemError, "WriteChar accepted something that is not a string");
        return NULL;
    }
    return Py_BuildValue("(NN)", read, pending());
}

/* A count of nothing names no buffer, so neither builder needs one. */
static PyObject *u_from_nothing(PyObject *self, PyObject *arg)
{
    (void)self;
    (void)arg;
    PyObject *wide = PyUnicode_FromWideChar(NULL, 0);
    if (wide == NULL) {
        return NULL;
    }
    PyObject *rows = PyList_New(0);
    if (rows == NULL || PyList_Append(rows, wide) < 0) {
        Py_DECREF(wide);
        Py_XDECREF(rows);
        return NULL;
    }
    Py_DECREF(wide);
    int kinds[] = {PyUnicode_1BYTE_KIND, PyUnicode_2BYTE_KIND, PyUnicode_4BYTE_KIND};
    for (size_t index = 0; index < sizeof(kinds) / sizeof(kinds[0]); index++) {
        PyObject *text = PyUnicode_FromKindAndData(kinds[index], NULL, 0);
        if (text == NULL || PyList_Append(rows, text) < 0) {
            Py_XDECREF(text);
            Py_DECREF(rows);
            return NULL;
        }
        Py_DECREF(text);
    }
    /* The kind is still checked, a count of nothing or not. */
    if (PyUnicode_FromKindAndData(3, NULL, 0) != NULL || !PyErr_Occurred()) {
        Py_DECREF(rows);
        PyErr_SetString(PyExc_SystemError, "PyUnicode_FromKindAndData accepted kind 3");
        return NULL;
    }
    return Py_BuildValue("(NN)", rows, pending());
}

/* An empty string still allocates, and reads back as ''. */
static PyObject *u_empty(PyObject *self, PyObject *arg)
{
    (void)self;
    (void)arg;
    return PyUnicode_New(0, 0);
}

/* `size` copies of `fill`, written through the canonical representation. */
static PyObject *filled(Py_ssize_t size, Py_UCS4 fill)
{
    PyObject *out = PyUnicode_New(size, fill);
    if (out == NULL) {
        return NULL;
    }
    int kind = PyUnicode_KIND(out);
    void *data = PyUnicode_DATA(out);
    for (Py_ssize_t index = 0; index < size; index++) {
        PyUnicode_WRITE(kind, data, index, fill);
    }
    return out;
}

/* A new string handed to another entry point instead of being returned: two of
   them at once, so the second conversion happens while the first is live. */
static PyObject *u_pairs(PyObject *self, PyObject *arg)
{
    (void)self;
    (void)arg;
    PyObject *dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }
    /* The type tests and the length are answered while the string is still
       being written, so asking them does not decide its contents early. */
    PyObject *key = PyUnicode_New(2, 'k');
    if (key != NULL
        && (!PyUnicode_Check(key) || !PyUnicode_CheckExact(key)
            || PyUnicode_GET_LENGTH(key) != 2)) {
        Py_DECREF(key);
        Py_DECREF(dict);
        PyErr_SetString(PyExc_SystemError, "a string being filled reads as something else");
        return NULL;
    }
    if (key != NULL) {
        int kind = PyUnicode_KIND(key);
        void *data = PyUnicode_DATA(key);
        PyUnicode_WRITE(kind, data, 0, 'k');
        PyUnicode_WRITE(kind, data, 1, 'k');
    }
    PyObject *value = filled(3, 0x3042);
    if (key == NULL || value == NULL) {
        Py_XDECREF(key);
        Py_XDECREF(value);
        Py_DECREF(dict);
        return NULL;
    }
    int failed = PyDict_SetItem(dict, key, value);
    Py_DECREF(key);
    Py_DECREF(value);
    if (failed < 0) {
        Py_DECREF(dict);
        return NULL;
    }
    return dict;
}

/* The same, as an operand of a binary operation. */
static PyObject *u_join(PyObject *self, PyObject *arg)
{
    (void)self;
    PyObject *tail = filled(2, 0x1f363);
    if (tail == NULL) {
        return NULL;
    }
    PyObject *joined = PyNumber_Add(arg, tail);
    Py_DECREF(tail);
    return joined;
}

static PyMethodDef methods[] = {
    {"pairs", u_pairs, METH_NOARGS, "a new string as a dict key and value"},
    {"join", u_join, METH_O, "a new string as the right operand of +"},
    {"escape", u_escape, METH_O, "escape '<' through the canonical representation"},
    {"shape", u_shape, METH_O, "the kind/ascii/maxchar/length a string reports"},
    {"first_point", u_first_point, METH_O, "the first code point, read through the typed data"},
    {"reverse", u_reverse, METH_O, "reverse through PyUnicode_ReadChar/WriteChar"},
    {"out_of_range", u_out_of_range, METH_NOARGS, "the index checks of the char accessors"},
    {"rejects", u_rejects, METH_NOARGS, "the argument checks of PyUnicode_New"},
    {"empty", u_empty, METH_NOARGS, "PyUnicode_New(0, 0)"},
    {"new_and_write", u_new_and_write, METH_VARARGS, "PyUnicode_New then PyUnicode_WriteChar"},
    {"not_a_string", u_not_a_string, METH_O, "the type checks of the char accessors"},
    {"from_nothing", u_from_nothing, METH_NOARGS, "the builders handed a count of nothing"},
    {NULL, NULL, 0, NULL},
};

static int exec_module(PyObject *module)
{
    if (PyModule_AddIntConstant(module, "ONE_BYTE", PyUnicode_1BYTE_KIND) < 0) {
        return -1;
    }
    if (PyModule_AddIntConstant(module, "TWO_BYTE", PyUnicode_2BYTE_KIND) < 0) {
        return -1;
    }
    if (PyModule_AddIntConstant(module, "FOUR_BYTE", PyUnicode_4BYTE_KIND) < 0) {
        return -1;
    }
    return 0;
}

static PyModuleDef_Slot slots[] = {
    {Py_mod_exec, exec_module},
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpyext_unicode",
    "pyre cpyext unicode module",
    0,
    methods,
    slots,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_cpyext_unicode(void)
{
    return PyModuleDef_Init(&moduledef);
}
