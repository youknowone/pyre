/* The conversions an extension does at its C boundary: the named codecs, the
   `wchar_t` forms, the filesystem encoding, and the small entry points beside
   them. */

#include <Python.h>

#include <stdlib.h>
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

/* What an entry point answering an object left behind: the object, or the
   exception it recorded instead. */
static PyObject *outcome(PyObject *answer)
{
    if (answer == NULL) {
        return pending();
    }
    PyObject *pair = Py_BuildValue("(OO)", answer, Py_None);
    Py_DECREF(answer);
    return pair;
}

/* The error handler name, which every codec entry point takes as NULL for
   `strict`. */
static const char *handler(PyObject *errors)
{
    return errors == Py_None ? NULL : PyUnicode_AsUTF8(errors);
}

/* ── the named codecs ─────────────────────────────────────────────────── */

static PyObject *decode_ascii(PyObject *self, PyObject *args)
{
    (void)self;
    const char *data;
    Py_ssize_t length;
    PyObject *errors;
    if (!PyArg_ParseTuple(args, "y#O", &data, &length, &errors)) {
        return NULL;
    }
    return outcome(PyUnicode_DecodeASCII(data, length, handler(errors)));
}

static PyObject *decode_latin1(PyObject *self, PyObject *args)
{
    (void)self;
    const char *data;
    Py_ssize_t length;
    PyObject *errors;
    if (!PyArg_ParseTuple(args, "y#O", &data, &length, &errors)) {
        return NULL;
    }
    return outcome(PyUnicode_DecodeLatin1(data, length, handler(errors)));
}

static PyObject *decode_named(PyObject *self, PyObject *args)
{
    (void)self;
    const char *data, *encoding;
    Py_ssize_t length;
    PyObject *errors;
    if (!PyArg_ParseTuple(args, "y#sO", &data, &length, &encoding, &errors)) {
        return NULL;
    }
    return outcome(PyUnicode_Decode(data, length, encoding, handler(errors)));
}

static PyObject *as_ascii(PyObject *self, PyObject *object)
{
    (void)self;
    return outcome(PyUnicode_AsASCIIString(object));
}

static PyObject *as_latin1(PyObject *self, PyObject *object)
{
    (void)self;
    return outcome(PyUnicode_AsLatin1String(object));
}

static PyObject *as_utf8_string(PyObject *self, PyObject *object)
{
    (void)self;
    return outcome(PyUnicode_AsUTF8String(object));
}

static PyObject *as_encoded(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *object, *errors;
    const char *encoding;
    if (!PyArg_ParseTuple(args, "OsO", &object, &encoding, &errors)) {
        return NULL;
    }
    return outcome(PyUnicode_AsEncodedString(object, encoding, handler(errors)));
}

/* ── the `wchar_t` forms ──────────────────────────────────────────────── */

/* `PyUnicode_FromWideChar` over a buffer built from the code points in
   `points`.  A `size` of -1 asks the entry point to find the NUL itself, so
   the buffer carries one either way. */
static PyObject *from_wide(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *points;
    Py_ssize_t size;
    if (!PyArg_ParseTuple(args, "On", &points, &size)) {
        return NULL;
    }
    Py_ssize_t count = PyList_Size(points);
    if (count < 0) {
        return NULL;
    }
    wchar_t *buffer = PyMem_New(wchar_t, count + 1);
    if (buffer == NULL) {
        return PyErr_NoMemory();
    }
    for (Py_ssize_t index = 0; index < count; index++) {
        long point = PyLong_AsLong(PyList_GetItem(points, index));
        if (point == -1 && PyErr_Occurred()) {
            PyMem_Free(buffer);
            return NULL;
        }
        buffer[index] = (wchar_t)point;
    }
    buffer[count] = 0;
    PyObject *answer = outcome(PyUnicode_FromWideChar(buffer, size));
    PyMem_Free(buffer);
    return answer;
}

/* `PyUnicode_AsWideChar` into a buffer of `room` slots: the count it answered
   and the slots it wrote, so a caller can see the NUL it appends when there is
   space and the truncation when there is not. */
static PyObject *as_wide(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *object;
    Py_ssize_t room;
    if (!PyArg_ParseTuple(args, "On", &object, &room)) {
        return NULL;
    }
    wchar_t *buffer = PyMem_New(wchar_t, room > 0 ? room : 1);
    if (buffer == NULL) {
        return PyErr_NoMemory();
    }
    memset(buffer, 0x7f, sizeof(wchar_t) * (size_t)(room > 0 ? room : 1));
    Py_ssize_t written = PyUnicode_AsWideChar(object, buffer, room);
    if (written < 0) {
        PyMem_Free(buffer);
        return pending();
    }
    PyObject *slots = PyList_New(room);
    if (slots == NULL) {
        PyMem_Free(buffer);
        return NULL;
    }
    for (Py_ssize_t index = 0; index < room; index++) {
        PyObject *point = PyLong_FromLong((long)buffer[index]);
        if (point == NULL) {
            Py_DECREF(slots);
            PyMem_Free(buffer);
            return NULL;
        }
        PyList_SET_ITEM(slots, index, point);
    }
    PyMem_Free(buffer);
    PyObject *answer = Py_BuildValue("(nO)", written, slots);
    Py_DECREF(slots);
    return answer;
}

/* `PyUnicode_AsWideCharString`: the length it reported and the code points of
   the block it allocated, read up to and including its NUL. */
static PyObject *as_wide_string(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *object;
    int want_size;
    if (!PyArg_ParseTuple(args, "Op", &object, &want_size)) {
        return NULL;
    }
    Py_ssize_t size = -2;
    wchar_t *block = PyUnicode_AsWideCharString(object, want_size ? &size : NULL);
    if (block == NULL) {
        return pending();
    }
    Py_ssize_t length = 0;
    while (block[length] != 0) {
        length++;
    }
    PyObject *points = PyList_New(length);
    if (points == NULL) {
        PyMem_Free(block);
        return NULL;
    }
    for (Py_ssize_t index = 0; index < length; index++) {
        PyObject *point = PyLong_FromLong((long)block[index]);
        if (point == NULL) {
            Py_DECREF(points);
            PyMem_Free(block);
            return NULL;
        }
        PyList_SET_ITEM(points, index, point);
    }
    PyMem_Free(block);
    PyObject *answer = Py_BuildValue("(nO)", size, points);
    Py_DECREF(points);
    return answer;
}

/* ── the filesystem encoding ──────────────────────────────────────────── */

static PyObject *decode_fs(PyObject *self, PyObject *args)
{
    (void)self;
    const char *data;
    if (!PyArg_ParseTuple(args, "y", &data)) {
        return NULL;
    }
    return outcome(PyUnicode_DecodeFSDefault(data));
}

static PyObject *decode_fs_size(PyObject *self, PyObject *args)
{
    (void)self;
    const char *data;
    Py_ssize_t length;
    if (!PyArg_ParseTuple(args, "y#", &data, &length)) {
        return NULL;
    }
    return outcome(PyUnicode_DecodeFSDefaultAndSize(data, length));
}

static PyObject *encode_fs(PyObject *self, PyObject *object)
{
    (void)self;
    return outcome(PyUnicode_EncodeFSDefault(object));
}

/* Both filesystem converters reached the way an extension reaches them: as an
   `O&` unit.  The second call is the release one the protocol asks for. */
static PyObject *fs_converter(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *converted = NULL;
    if (!PyArg_ParseTuple(args, "O&", PyUnicode_FSConverter, &converted)) {
        return pending();
    }
    PyObject *answer = Py_BuildValue("(OO)", converted, Py_None);
    Py_DECREF(converted);
    return answer;
}

static PyObject *fs_decoder(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *converted = NULL;
    if (!PyArg_ParseTuple(args, "O&", PyUnicode_FSDecoder, &converted)) {
        return pending();
    }
    PyObject *answer = Py_BuildValue("(OO)", converted, Py_None);
    Py_DECREF(converted);
    return answer;
}

/* ── the small entry points ───────────────────────────────────────────── */

static PyObject *index_check(PyObject *self, PyObject *object)
{
    (void)self;
    return PyLong_FromLong(PyIndex_Check(object));
}

static PyObject *get_constant(PyObject *self, PyObject *args)
{
    (void)self;
    unsigned int identifier;
    int borrowed;
    if (!PyArg_ParseTuple(args, "Ip", &identifier, &borrowed)) {
        return NULL;
    }
    if (borrowed) {
        PyObject *value = Py_GetConstantBorrowed(identifier);
        return outcome(value == NULL ? NULL : Py_NewRef(value));
    }
    return outcome(Py_GetConstant(identifier));
}

/* `PyOS_snprintf` into a buffer of `room` bytes: what it returned and the
   bytes it left, so truncation is visible. */
static PyObject *os_snprintf(PyObject *self, PyObject *args)
{
    (void)self;
    Py_ssize_t room;
    if (!PyArg_ParseTuple(args, "n", &room)) {
        return NULL;
    }
    char *buffer = PyMem_Malloc((size_t)room);
    if (buffer == NULL) {
        return PyErr_NoMemory();
    }
    memset(buffer, 'x', (size_t)room);
    int written = PyOS_snprintf(buffer, (size_t)room, "%s-%d-%s", "left", 42, "right");
    PyObject *answer = Py_BuildValue("(iy#)", written, buffer, room);
    PyMem_Free(buffer);
    return answer;
}

/* ── PyArg_Parse ──────────────────────────────────────────────────────── */

static PyObject *arg_parse(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *value;
    const char *format;
    if (!PyArg_ParseTuple(args, "Os", &value, &format)) {
        return NULL;
    }
    if (strcmp(format, "i") == 0) {
        int one = -1;
        int rc = PyArg_Parse(value, "i", &one);
        return Py_BuildValue("(iiO)", rc, one, pending());
    }
    if (strcmp(format, "(ii)") == 0) {
        int one = -1, two = -1;
        int rc = PyArg_Parse(value, "(ii)", &one, &two);
        return Py_BuildValue("(i(ii)O)", rc, one, two, pending());
    }
    if (strcmp(format, "ii") == 0) {
        int one = -1, two = -1;
        int rc = PyArg_Parse(value, "ii", &one, &two);
        return Py_BuildValue("(i(ii)O)", rc, one, two, pending());
    }
    if (strcmp(format, "s") == 0) {
        const char *text = NULL;
        int rc = PyArg_Parse(value, "s", &text);
        return Py_BuildValue("(isO)", rc, text, pending());
    }
    PyErr_SetString(PyExc_ValueError, "the fixture does not offer that format");
    return NULL;
}

/* A nested unit through the ordinary tuple parser, which is where an
   extension usually writes one. */
static PyObject *parse_nested(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *inner;
    const char *shape;
    if (!PyArg_ParseTuple(args, "Os", &inner, &shape)) {
        return NULL;
    }
    PyObject *one_tuple = PyTuple_Pack(1, inner);
    if (one_tuple == NULL) {
        return NULL;
    }
    int one = -1, two = -1;
    int rc;
    if (strcmp(shape, "(ii)") == 0) {
        rc = PyArg_ParseTuple(one_tuple, "(ii)", &one, &two);
    } else if (strcmp(shape, "(i)i") == 0) {
        PyObject *pair = PyTuple_Pack(2, inner, inner);
        Py_DECREF(one_tuple);
        if (pair == NULL) {
            return NULL;
        }
        rc = PyArg_ParseTuple(pair, "(i)O", &one, &inner);
        Py_DECREF(pair);
        return Py_BuildValue("(i(ii)O)", rc, one, two, pending());
    } else {
        Py_DECREF(one_tuple);
        PyErr_SetString(PyExc_ValueError, "the fixture does not offer that shape");
        return NULL;
    }
    Py_DECREF(one_tuple);
    return Py_BuildValue("(i(ii)O)", rc, one, two, pending());
}

static PyMethodDef methods[] = {
    {"decode_ascii", decode_ascii, METH_VARARGS, NULL},
    {"decode_latin1", decode_latin1, METH_VARARGS, NULL},
    {"decode_named", decode_named, METH_VARARGS, NULL},
    {"as_ascii", as_ascii, METH_O, NULL},
    {"as_latin1", as_latin1, METH_O, NULL},
    {"as_utf8_string", as_utf8_string, METH_O, NULL},
    {"as_encoded", as_encoded, METH_VARARGS, NULL},
    {"from_wide", from_wide, METH_VARARGS, NULL},
    {"as_wide", as_wide, METH_VARARGS, NULL},
    {"as_wide_string", as_wide_string, METH_VARARGS, NULL},
    {"decode_fs", decode_fs, METH_VARARGS, NULL},
    {"decode_fs_size", decode_fs_size, METH_VARARGS, NULL},
    {"encode_fs", encode_fs, METH_O, NULL},
    {"fs_converter", fs_converter, METH_VARARGS, NULL},
    {"fs_decoder", fs_decoder, METH_VARARGS, NULL},
    {"index_check", index_check, METH_O, NULL},
    {"get_constant", get_constant, METH_VARARGS, NULL},
    {"os_snprintf", os_snprintf, METH_VARARGS, NULL},
    {"arg_parse", arg_parse, METH_VARARGS, NULL},
    {"parse_nested", parse_nested, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_conversions", NULL, -1,
                                 methods};

PyMODINIT_FUNC PyInit_cpyext_conversions(void)
{
    PyObject *module = PyModule_Create(&def);
    if (module == NULL) {
        return NULL;
    }
    if (PyModule_AddIntConstant(module, "WCHAR_SIZE", (long)sizeof(wchar_t)) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
