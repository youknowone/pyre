/* The `str` an extension builds piece by piece, and the container entry
   points that answer "was it there?" alongside the value. */

#include <Python.h>

#include <string.h>
#include <wchar.h>

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

/* ── the writer ───────────────────────────────────────────────────────── */

/* Everything the writer can be handed, in one pass: the answer is the string
   it built, or what it refused to build. */
static PyObject *write_all(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *value;
    if (!PyArg_ParseTuple(args, "O", &value)) {
        return NULL;
    }
    PyUnicodeWriter *writer = PyUnicodeWriter_Create(0);
    if (writer == NULL) {
        return pending();
    }
    Py_UCS4 points[] = {0x61, 0x1F600, 0x62};
    wchar_t wide[] = {0x41, 0x2603, 0};
    if (PyUnicodeWriter_WriteASCII(writer, "[", 1) < 0
        || PyUnicodeWriter_WriteUTF8(writer, "ol\xc3\xa9", -1) < 0
        || PyUnicodeWriter_WriteChar(writer, 0x2603) < 0
        || PyUnicodeWriter_WriteUCS4(writer, points, 3) < 0
        || PyUnicodeWriter_WriteWideChar(writer, wide, -1) < 0
        || PyUnicodeWriter_WriteStr(writer, value) < 0
        || PyUnicodeWriter_WriteRepr(writer, value) < 0
        || PyUnicodeWriter_Format(writer, "<%s %i %V>", "fmt", 7, NULL, "v") < 0
        || PyUnicodeWriter_WriteASCII(writer, "]", -1) < 0) {
        PyUnicodeWriter_Discard(writer);
        return pending();
    }
    return outcome(PyUnicodeWriter_Finish(writer));
}

/* A writer given up without being asked for its string. */
static PyObject *write_discard(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    PyUnicodeWriter *writer = PyUnicodeWriter_Create(16);
    if (writer == NULL) {
        return pending();
    }
    int written = PyUnicodeWriter_WriteASCII(writer, "gone", 4);
    PyUnicodeWriter_Discard(writer);
    return Py_BuildValue("(iN)", written, pending());
}

/* `PyUnicodeWriter_Create` with a length it will not take. */
static PyObject *write_bad_create(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    PyUnicodeWriter *writer = PyUnicodeWriter_Create(-1);
    if (writer != NULL) {
        PyUnicodeWriter_Discard(writer);
        return Py_BuildValue("(sO)", "accepted", Py_None);
    }
    return pending();
}

/* Each refusal, named, with whatever the writer still held afterwards. */
static PyObject *write_refusals(PyObject *self, PyObject *args)
{
    (void)self;
    const char *which;
    PyObject *value;
    Py_ssize_t start;
    Py_ssize_t end;
    if (!PyArg_ParseTuple(args, "sOnn", &which, &value, &start, &end)) {
        return NULL;
    }
    PyUnicodeWriter *writer = PyUnicodeWriter_Create(0);
    if (writer == NULL) {
        return pending();
    }
    if (PyUnicodeWriter_WriteASCII(writer, "keep", 4) < 0) {
        PyUnicodeWriter_Discard(writer);
        return pending();
    }
    int written;
    if (strcmp(which, "char") == 0) {
        written = PyUnicodeWriter_WriteChar(writer, 0x110000);
    } else if (strcmp(which, "utf8") == 0) {
        written = PyUnicodeWriter_WriteUTF8(writer, "a\xff\x62", 3);
    } else if (strcmp(which, "utf8_short") == 0) {
        written = PyUnicodeWriter_WriteUTF8(writer, "a\xc3", 2);
    } else if (strcmp(which, "ucs4") == 0) {
        Py_UCS4 points[] = {0x61};
        written = PyUnicodeWriter_WriteUCS4(writer, points, -1);
    } else if (strcmp(which, "substring") == 0) {
        written = PyUnicodeWriter_WriteSubstring(writer, value, start, end);
    } else if (strcmp(which, "str") == 0) {
        written = PyUnicodeWriter_WriteStr(writer, value);
    } else if (strcmp(which, "repr") == 0) {
        written = PyUnicodeWriter_WriteRepr(writer, value);
    } else {
        PyUnicodeWriter_Discard(writer);
        return PyErr_Format(PyExc_ValueError, "no such case: %s", which);
    }
    PyObject *left = pending();
    PyObject *held = PyUnicodeWriter_Finish(writer);
    if (held == NULL) {
        Py_DECREF(left);
        return pending();
    }
    return Py_BuildValue("(iNN)", written, left, held);
}

/* The substring cases that work. */
static PyObject *write_substring(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *value;
    Py_ssize_t start;
    Py_ssize_t end;
    if (!PyArg_ParseTuple(args, "Onn", &value, &start, &end)) {
        return NULL;
    }
    PyUnicodeWriter *writer = PyUnicodeWriter_Create(0);
    if (writer == NULL) {
        return pending();
    }
    if (PyUnicodeWriter_WriteSubstring(writer, value, start, end) < 0) {
        PyUnicodeWriter_Discard(writer);
        return pending();
    }
    return outcome(PyUnicodeWriter_Finish(writer));
}

/* ── dict.pop ─────────────────────────────────────────────────────────── */

static PyObject *dict_pop(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *mapping;
    PyObject *key;
    if (!PyArg_ParseTuple(args, "OO", &mapping, &key)) {
        return NULL;
    }
    PyObject *value = (PyObject *)0x1;
    int answer = PyDict_Pop(mapping, key, &value);
    if (value == NULL) {
        value = Py_NewRef(Py_None);
    }
    return Py_BuildValue("(iNN)", answer, value, pending());
}

static PyObject *dict_pop_no_result(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *mapping;
    PyObject *key;
    if (!PyArg_ParseTuple(args, "OO", &mapping, &key)) {
        return NULL;
    }
    int answer = PyDict_Pop(mapping, key, NULL);
    return Py_BuildValue("(iN)", answer, pending());
}

static PyObject *dict_pop_string(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *mapping;
    const char *key;
    if (!PyArg_ParseTuple(args, "Os", &mapping, &key)) {
        return NULL;
    }
    PyObject *value = (PyObject *)0x1;
    int answer = PyDict_PopString(mapping, key, &value);
    if (value == NULL) {
        value = Py_NewRef(Py_None);
    }
    return Py_BuildValue("(iNN)", answer, value, pending());
}

static PyObject *dict_proxy(PyObject *self, PyObject *value)
{
    (void)self;
    return outcome(PyDictProxy_New(value));
}

/* ── the optional mapping read ────────────────────────────────────────── */

static PyObject *optional_item(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *mapping;
    PyObject *key;
    if (!PyArg_ParseTuple(args, "OO", &mapping, &key)) {
        return NULL;
    }
    PyObject *value = (PyObject *)0x1;
    int answer = PyMapping_GetOptionalItem(mapping, key, &value);
    if (value == NULL) {
        value = Py_NewRef(Py_None);
    }
    return Py_BuildValue("(iNN)", answer, value, pending());
}

static PyObject *optional_item_string(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *mapping;
    const char *key;
    if (!PyArg_ParseTuple(args, "Os", &mapping, &key)) {
        return NULL;
    }
    PyObject *value = (PyObject *)0x1;
    int answer = PyMapping_GetOptionalItemString(mapping, key, &value);
    if (value == NULL) {
        value = Py_NewRef(Py_None);
    }
    return Py_BuildValue("(iNN)", answer, value, pending());
}

/* ── list.clear and list.extend ───────────────────────────────────────── */

static PyObject *list_clear(PyObject *self, PyObject *value)
{
    (void)self;
    int answer = PyList_Clear(value);
    return Py_BuildValue("(iN)", answer, pending());
}

static PyObject *list_extend(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *list;
    PyObject *iterable;
    if (!PyArg_ParseTuple(args, "OO", &list, &iterable)) {
        return NULL;
    }
    int answer = PyList_Extend(list, iterable);
    return Py_BuildValue("(iN)", answer, pending());
}

/* ── bytes.join and bytes concatenation ───────────────────────────────── */

static PyObject *bytes_join(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *separator;
    PyObject *iterable;
    if (!PyArg_ParseTuple(args, "OO", &separator, &iterable)) {
        return NULL;
    }
    return outcome(PyBytes_Join(separator, iterable));
}

static PyObject *bytes_concat(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *left;
    PyObject *right;
    if (!PyArg_ParseTuple(args, "OO", &left, &right)) {
        return NULL;
    }
    Py_INCREF(left);
    PyBytes_Concat(&left, right == Py_None ? NULL : right);
    PyObject *answer = Py_BuildValue("(ON)", left == NULL ? Py_None : left,
                                     pending());
    Py_XDECREF(left);
    return answer;
}

static PyObject *bytes_concat_and_del(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *left;
    PyObject *right;
    if (!PyArg_ParseTuple(args, "OO", &left, &right)) {
        return NULL;
    }
    Py_INCREF(left);
    Py_INCREF(right);
    PyBytes_ConcatAndDel(&left, right);
    PyObject *answer = Py_BuildValue("(ON)", left == NULL ? Py_None : left,
                                     pending());
    Py_XDECREF(left);
    return answer;
}

static PyMethodDef methods[] = {
    {"write_all", write_all, METH_VARARGS, NULL},
    {"write_discard", write_discard, METH_NOARGS, NULL},
    {"write_bad_create", write_bad_create, METH_NOARGS, NULL},
    {"write_refusals", write_refusals, METH_VARARGS, NULL},
    {"write_substring", write_substring, METH_VARARGS, NULL},
    {"dict_pop", dict_pop, METH_VARARGS, NULL},
    {"dict_pop_no_result", dict_pop_no_result, METH_VARARGS, NULL},
    {"dict_pop_string", dict_pop_string, METH_VARARGS, NULL},
    {"dict_proxy", dict_proxy, METH_O, NULL},
    {"optional_item", optional_item, METH_VARARGS, NULL},
    {"optional_item_string", optional_item_string, METH_VARARGS, NULL},
    {"list_clear", list_clear, METH_O, NULL},
    {"list_extend", list_extend, METH_VARARGS, NULL},
    {"bytes_join", bytes_join, METH_VARARGS, NULL},
    {"bytes_concat", bytes_concat, METH_VARARGS, NULL},
    {"bytes_concat_and_del", bytes_concat_and_del, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_writer", NULL,
                                 -1, methods};

PyMODINIT_FUNC PyInit_cpyext_writer(void)
{
    return PyModule_Create(&def);
}
