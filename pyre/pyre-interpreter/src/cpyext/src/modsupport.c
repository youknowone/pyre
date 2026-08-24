/* `Py_BuildValue` and the unit walk behind it.
 *
 * Compiled into the interpreter and exported from it, which is where
 * `pypy/module/cpyext/src/modsupport.c` puts the same bodies.  Everything here is
 * built out of the non-variadic entry points `pyre_decl.h` declares, and the
 * headers carry the declarations alone -- an extension resolves these at load
 * time rather than compiling a copy of each.
 */
#include "Python.h"

static PyObject *_PyPyre_BuildValue(const char **format, va_list *va);

/* One `Py_BuildValue` unit.  Containers recurse until their closing bracket. */
static PyObject *_PyPyre_BuildOne(const char **format, va_list *va)
{
    char code = **format;
    (*format)++;
    switch (code) {
    case 'i': case 'b': case 'h':
        return PyLong_FromLong((long)va_arg(*va, int));
    case 'B': case 'H': case 'I':
        return PyLong_FromUnsignedLong((unsigned long)va_arg(*va, unsigned int));
    case 'l':
        return PyLong_FromLong(va_arg(*va, long));
    case 'k':
        return PyLong_FromUnsignedLong(va_arg(*va, unsigned long));
    case 'L':
        return PyLong_FromLongLong(va_arg(*va, long long));
    case 'K':
        return PyLong_FromUnsignedLongLong(va_arg(*va, unsigned long long));
    case 'n':
        return PyLong_FromSsize_t(va_arg(*va, Py_ssize_t));
    case 'f': case 'd':
        return PyFloat_FromDouble(va_arg(*va, double));
    case 'c': {
        char value = (char)va_arg(*va, int);
        return PyBytes_FromStringAndSize(&value, 1);
    }
    case 'C': {
        char value = (char)va_arg(*va, int);
        return PyUnicode_FromStringAndSize(&value, 1);
    }
    case 's': case 'z': {
        const char *text = va_arg(*va, const char *);
        Py_ssize_t length = -1;
        if (**format == '#') {
            (*format)++;
            length = va_arg(*va, Py_ssize_t);
        }
        if (text == NULL) {
            Py_INCREF(Py_None);
            return Py_None;
        }
        return length < 0 ? PyUnicode_FromString(text)
                          : PyUnicode_FromStringAndSize(text, length);
    }
    case 'y': {
        const char *text = va_arg(*va, const char *);
        Py_ssize_t length = -1;
        if (**format == '#') {
            (*format)++;
            length = va_arg(*va, Py_ssize_t);
        }
        if (text == NULL) {
            Py_INCREF(Py_None);
            return Py_None;
        }
        return length < 0 ? PyBytes_FromString(text)
                          : PyBytes_FromStringAndSize(text, length);
    }
    case 'O': case 'S': case 'N': {
        PyObject *value = va_arg(*va, PyObject *);
        if (value == NULL) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_SystemError,
                                "Py_BuildValue: NULL object passed to O format");
            }
            return NULL;
        }
        if (code != 'N') {
            Py_INCREF(value);
        }
        return value;
    }
    case '(': case '[': {
        char closing = code == '(' ? ')' : ']';
        PyObject *items = PyList_New(0);
        if (items == NULL) {
            return NULL;
        }
        while (**format && **format != closing) {
            if (**format == ',' || **format == ' ') {
                (*format)++;
                continue;
            }
            PyObject *item = _PyPyre_BuildOne(format, va);
            if (item == NULL || PyList_Append(items, item) < 0) {
                Py_XDECREF(item);
                Py_DECREF(items);
                return NULL;
            }
            Py_DECREF(item);
        }
        if (**format == closing) {
            (*format)++;
        }
        if (closing == ']') {
            return items;
        }
        Py_ssize_t size = PyList_Size(items);
        PyObject *tuple = PyTuple_New(size);
        if (tuple == NULL) {
            Py_DECREF(items);
            return NULL;
        }
        for (Py_ssize_t index = 0; index < size; index++) {
            PyObject *item = PyList_GetItem(items, index);
            Py_INCREF(item);
            PyTuple_SetItem(tuple, index, item);
        }
        Py_DECREF(items);
        return tuple;
    }
    case '{': {
        PyObject *mapping = PyDict_New();
        if (mapping == NULL) {
            return NULL;
        }
        while (**format && **format != '}') {
            if (**format == ',' || **format == ' ' || **format == ':') {
                (*format)++;
                continue;
            }
            PyObject *key = _PyPyre_BuildOne(format, va);
            if (key == NULL) {
                Py_DECREF(mapping);
                return NULL;
            }
            while (**format == ',' || **format == ' ' || **format == ':') {
                (*format)++;
            }
            PyObject *value = _PyPyre_BuildOne(format, va);
            if (value == NULL || PyDict_SetItem(mapping, key, value) < 0) {
                Py_DECREF(key);
                Py_XDECREF(value);
                Py_DECREF(mapping);
                return NULL;
            }
            Py_DECREF(key);
            Py_DECREF(value);
        }
        if (**format == '}') {
            (*format)++;
        }
        return mapping;
    }
    default:
        PyErr_SetString(PyExc_SystemError, "Py_BuildValue: unsupported format");
        return NULL;
    }
}

static PyObject *_PyPyre_BuildValue(const char **format, va_list *va)
{
    while (**format == ' ' || **format == ',') {
        (*format)++;
    }
    if (**format == '\0') {
        Py_INCREF(Py_None);
        return Py_None;
    }
    const char *lookahead = *format;
    PyObject *first = _PyPyre_BuildOne(format, va);
    if (first == NULL) {
        return NULL;
    }
    while (**format == ' ' || **format == ',') {
        (*format)++;
    }
    if (**format == '\0') {
        return first;
    }
    (void)lookahead;
    /* More than one unit builds a tuple, exactly as `Py_BuildValue`
       documents. */
    PyObject *items = PyList_New(0);
    if (items == NULL || PyList_Append(items, first) < 0) {
        Py_DECREF(first);
        Py_XDECREF(items);
        return NULL;
    }
    Py_DECREF(first);
    while (**format) {
        if (**format == ' ' || **format == ',') {
            (*format)++;
            continue;
        }
        PyObject *item = _PyPyre_BuildOne(format, va);
        if (item == NULL || PyList_Append(items, item) < 0) {
            Py_XDECREF(item);
            Py_DECREF(items);
            return NULL;
        }
        Py_DECREF(item);
    }
    Py_ssize_t size = PyList_Size(items);
    PyObject *tuple = PyTuple_New(size);
    if (tuple == NULL) {
        Py_DECREF(items);
        return NULL;
    }
    for (Py_ssize_t index = 0; index < size; index++) {
        PyObject *item = PyList_GetItem(items, index);
        Py_INCREF(item);
        PyTuple_SetItem(tuple, index, item);
    }
    Py_DECREF(items);
    return tuple;
}

PyObject *Py_BuildValue(const char *format, ...)
{
    va_list va;
    va_start(va, format);
    const char *cursor = format;
    PyObject *value = _PyPyre_BuildValue(&cursor, &va);
    va_end(va);
    return value;
}

/* The same build over a list the caller already opened, which is how a
   variadic entry point of its own passes its arguments on.

   The walk goes over a `va_copy`, not over `&va`: an argument of type
   `va_list` is an array on some ABIs, where it decays to a pointer and `&va`
   names the parameter slot rather than the list.  Copying gives an object of
   the type the walker takes the address of, on every ABI, and leaves the
   caller's own list where it was. */
PyObject *Py_VaBuildValue(const char *format, va_list va)
{
    va_list copy;
    va_copy(copy, va);
    const char *cursor = format;
    PyObject *value = _PyPyre_BuildValue(&cursor, &copy);
    va_end(copy);
    return value;
}
