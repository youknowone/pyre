/* A multi-phase (PEP 489) extension exercising the method calling
   conventions, the argument parsers, the object constructors and the
   exception indicator. */

#include <Python.h>
#include <string.h>

typedef struct {
    long calls;
} methods_state;

static struct PyModuleDef moduledef;

static methods_state *state_of(PyObject *module)
{
    methods_state *state = (methods_state *)PyModule_GetState(module);
    if (state == NULL && !PyErr_Occurred()) {
        PyErr_SetString(PyExc_SystemError, "module state is missing");
    }
    return state;
}

/* METH_NOARGS: the second argument is NULL and `self` is the module. */
static PyObject *m_bump(PyObject *self, PyObject *unused)
{
    if (unused != NULL) {
        PyErr_SetString(PyExc_SystemError, "METH_NOARGS was passed an argument");
        return NULL;
    }
    methods_state *state = state_of(self);
    if (state == NULL) {
        return NULL;
    }
    state->calls++;
    return PyLong_FromLong(state->calls);
}

/* METH_O */
static PyObject *m_wrap(PyObject *self, PyObject *arg)
{
    (void)self;
    return Py_BuildValue("(Os)", arg, "seen");
}

/* METH_VARARGS with an optional unit. */
static PyObject *m_add(PyObject *self, PyObject *args)
{
    (void)self;
    long left = 0;
    long right = 10;
    if (!PyArg_ParseTuple(args, "l|l:add", &left, &right)) {
        return NULL;
    }
    return PyLong_FromLong(left + right);
}

/* METH_VARARGS | METH_KEYWORDS */
static PyObject *m_greet(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static char *keywords[] = {"name", "punct", NULL};
    const char *name = NULL;
    const char *punct = "!";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|s:greet", keywords,
                                     &name, &punct)) {
        return NULL;
    }
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "hello %s%s", name, punct);
    return PyUnicode_FromString(buffer);
}

/* METH_FASTCALL */
static PyObject *m_total(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    (void)self;
    long total = 0;
    for (Py_ssize_t index = 0; index < nargs; index++) {
        long value = PyLong_AsLong(args[index]);
        if (value == -1 && PyErr_Occurred()) {
            return NULL;
        }
        total += value;
    }
    return PyLong_FromLong(total);
}

/* METH_FASTCALL | METH_KEYWORDS: the keyword values follow the positional
   ones in `args`, and `kwnames` names them in order. */
static PyObject *m_layout(PyObject *self, PyObject *const *args,
                          Py_ssize_t nargs, PyObject *kwnames)
{
    (void)self;
    PyObject *positional = PyList_New(0);
    if (positional == NULL) {
        return NULL;
    }
    for (Py_ssize_t index = 0; index < nargs; index++) {
        if (PyList_Append(positional, args[index]) < 0) {
            Py_DECREF(positional);
            return NULL;
        }
    }
    PyObject *named = PyList_New(0);
    if (named == NULL) {
        Py_DECREF(positional);
        return NULL;
    }
    Py_ssize_t count = kwnames == NULL ? 0 : PyTuple_Size(kwnames);
    for (Py_ssize_t index = 0; index < count; index++) {
        PyObject *pair = Py_BuildValue("(OO)", PyTuple_GetItem(kwnames, index),
                                       args[nargs + index]);
        if (pair == NULL || PyList_Append(named, pair) < 0) {
            Py_XDECREF(pair);
            Py_DECREF(named);
            Py_DECREF(positional);
            return NULL;
        }
        Py_DECREF(pair);
    }
    PyObject *result = Py_BuildValue("(NN)", positional, named);
    if (result == NULL) {
        Py_DECREF(named);
        Py_DECREF(positional);
    }
    return result;
}

/* PyArg_UnpackTuple and the object protocol. */
static PyObject *m_apply(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *callable = NULL;
    PyObject *argument = NULL;
    if (!PyArg_UnpackTuple(args, "apply", 2, 2, &callable, &argument)) {
        return NULL;
    }
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "apply() needs a callable");
        return NULL;
    }
    PyObject *call_args = Py_BuildValue("(O)", argument);
    if (call_args == NULL) {
        return NULL;
    }
    PyObject *result = PyObject_CallObject(callable, call_args);
    Py_DECREF(call_args);
    return result;
}

/* Attribute, item and stringification entry points. */
static PyObject *m_inspect(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *target = NULL;
    const char *name = NULL;
    if (!PyArg_ParseTuple(args, "Os:inspect", &target, &name)) {
        return NULL;
    }
    int present = PyObject_HasAttrString(target, name);
    PyObject *text = PyObject_Str(target);
    if (text == NULL) {
        return NULL;
    }
    PyObject *shown = PyObject_Repr(target);
    if (shown == NULL) {
        Py_DECREF(text);
        return NULL;
    }
    Py_ssize_t size = PyObject_Size(target);
    if (size < 0) {
        PyErr_Clear();
        size = -1;
    }
    PyObject *result = Py_BuildValue("(iNNni)", present, text, shown, size,
                                     PyObject_IsTrue(target));
    return result;
}

/* The primitive constructors and accessors. */
static PyObject *m_build(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    return Py_BuildValue("{s:i,s:l,s:d,s:s,s:y,s:[i,i,i],s:(s,i)}",
                         "int", 3,
                         "long", 4L,
                         "float", 1.5,
                         "str", "text",
                         "bytes", "raw",
                         "list", 1, 2, 3,
                         "tuple", "pair", 9);
}

static PyObject *m_roundtrip(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *text = NULL;
    PyObject *raw = NULL;
    if (!PyArg_ParseTuple(args, "US:roundtrip", &text, &raw)) {
        return NULL;
    }
    Py_ssize_t text_size = 0;
    const char *utf8 = PyUnicode_AsUTF8AndSize(text, &text_size);
    if (utf8 == NULL) {
        return NULL;
    }
    char *bytes = NULL;
    Py_ssize_t bytes_size = 0;
    if (PyBytes_AsStringAndSize(raw, &bytes, &bytes_size) < 0) {
        return NULL;
    }
    return Py_BuildValue("(s#y#nn)", utf8, text_size, bytes, bytes_size,
                         PyUnicode_GetLength(text), PyBytes_Size(raw));
}

static PyObject *m_numbers(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *value = NULL;
    if (!PyArg_ParseTuple(args, "O:numbers", &value)) {
        return NULL;
    }
    long as_long = PyLong_AsLong(value);
    if (as_long == -1 && PyErr_Occurred()) {
        return NULL;
    }
    double as_double = PyLong_AsDouble(value);
    if (as_double == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    PyObject *parsed = PyLong_FromString("-0x2a", NULL, 0);
    if (parsed == NULL) {
        return NULL;
    }
    return Py_BuildValue("(ldNiid)", as_long, as_double, parsed,
                         PyLong_Check(value), PyFloat_Check(value),
                         PyFloat_AsDouble(PyFloat_FromDouble(0.25)));
}

static PyObject *m_dict_ops(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    PyObject *mapping = PyDict_New();
    if (mapping == NULL) {
        return NULL;
    }
    PyObject *value = PyLong_FromLong(7);
    if (value == NULL || PyDict_SetItemString(mapping, "seven", value) < 0) {
        Py_XDECREF(value);
        Py_DECREF(mapping);
        return NULL;
    }
    Py_DECREF(value);
    PyObject *key = PyUnicode_FromString("seven");
    if (key == NULL) {
        Py_DECREF(mapping);
        return NULL;
    }
    PyObject *found = PyDict_GetItem(mapping, key);
    long seven = found == NULL ? -1 : PyLong_AsLong(found);
    int contains = PyDict_Contains(mapping, key);
    Py_ssize_t before = PyDict_Size(mapping);
    if (PyDict_DelItem(mapping, key) < 0) {
        Py_DECREF(key);
        Py_DECREF(mapping);
        return NULL;
    }
    Py_ssize_t after = PyDict_Size(mapping);
    PyObject *missing = PyDict_GetItemString(mapping, "seven");
    Py_DECREF(key);
    Py_DECREF(mapping);
    return Py_BuildValue("(linni)", seven, contains, before, after,
                         missing == NULL);
}

static PyObject *m_sequences(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    PyObject *tuple = PyTuple_New(2);
    if (tuple == NULL) {
        return NULL;
    }
    PyObject *first = PyLong_FromLong(1);
    if (first == NULL || PyTuple_SetItem(tuple, 0, first) < 0) {
        Py_DECREF(tuple);
        return NULL;
    }
    PyObject *second = PyUnicode_FromString("two");
    if (second == NULL || PyTuple_SetItem(tuple, 1, second) < 0) {
        Py_DECREF(tuple);
        return NULL;
    }
    PyObject *list = PyList_New(1);
    if (list == NULL) {
        Py_DECREF(tuple);
        return NULL;
    }
    PyObject *only = PyLong_FromLong(5);
    if (only == NULL || PyList_SetItem(list, 0, only) < 0) {
        Py_DECREF(list);
        Py_DECREF(tuple);
        return NULL;
    }
    PyObject *extra = PyBytes_FromStringAndSize("ab\0c", 4);
    if (extra == NULL || PyList_Append(list, extra) < 0) {
        Py_XDECREF(extra);
        Py_DECREF(list);
        Py_DECREF(tuple);
        return NULL;
    }
    Py_DECREF(extra);
    return Py_BuildValue("(NNnn)", tuple, list, PyTuple_Size(tuple),
                         PyList_Size(list));
}

/* The singletons, so the caller can check identity from Python. */
static PyObject *m_singletons(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    return Py_BuildValue("(OOOOO)", Py_None, Py_True, Py_False, Py_Ellipsis,
                         Py_NotImplemented);
}

static PyObject *m_predicates(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *value = NULL;
    if (!PyArg_ParseTuple(args, "O:predicates", &value)) {
        return NULL;
    }
    return Py_BuildValue("(iiiiiii)", Py_IsNone(value), Py_IsTrue(value),
                         PyUnicode_Check(value), PyBytes_Check(value),
                         PyTuple_Check(value), PyList_Check(value),
                         PyDict_Check(value));
}

/* The exception indicator. */
static PyObject *m_fail(PyObject *self, PyObject *args)
{
    (void)self;
    const char *kind = NULL;
    if (!PyArg_ParseTuple(args, "s:fail", &kind)) {
        return NULL;
    }
    if (strcmp(kind, "value") == 0) {
        PyErr_SetString(PyExc_ValueError, "a value complaint");
        return NULL;
    }
    if (strcmp(kind, "format") == 0) {
        PyErr_Format(PyExc_TypeError, "formatted %s and %d", "message", 7);
        return NULL;
    }
    if (strcmp(kind, "object") == 0) {
        PyObject *value = PyUnicode_FromString("payload");
        if (value == NULL) {
            return NULL;
        }
        PyErr_SetObject(PyExc_KeyError, value);
        Py_DECREF(value);
        return NULL;
    }
    if (strcmp(kind, "none") == 0) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    if (strcmp(kind, "memory") == 0) {
        return PyErr_NoMemory();
    }
    if (strcmp(kind, "argument") == 0) {
        PyErr_BadArgument();
        return NULL;
    }
    PyErr_SetString(PyExc_RuntimeError, "unknown failure kind");
    return NULL;
}

/* Catch a raised exception inside C: set, observe, match and clear. */
static PyObject *m_caught(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    PyErr_SetString(PyExc_IndexError, "swallowed");
    PyObject *occurred = PyErr_Occurred();
    int is_lookup = PyErr_ExceptionMatches(PyExc_LookupError);
    int is_type = PyErr_ExceptionMatches(PyExc_TypeError);
    PyObject *type = NULL;
    PyObject *value = NULL;
    PyObject *traceback = NULL;
    PyErr_Fetch(&type, &value, &traceback);
    int cleared = PyErr_Occurred() == NULL;
    PyObject *text = value == NULL ? PyUnicode_FromString("") : PyObject_Str(value);
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    if (text == NULL) {
        return NULL;
    }
    return Py_BuildValue("(iiiiN)", occurred != NULL, is_lookup, is_type,
                         cleared, text);
}

static PyMethodDef methods[] = {
    {"bump", (PyCFunction)m_bump, METH_NOARGS, "bump the module counter"},
    {"wrap", (PyCFunction)m_wrap, METH_O, NULL},
    {"add", (PyCFunction)m_add, METH_VARARGS, NULL},
    {"greet", (PyCFunction)(void (*)(void))m_greet,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"total", (PyCFunction)(void (*)(void))m_total, METH_FASTCALL, NULL},
    {"layout", (PyCFunction)(void (*)(void))m_layout,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"apply", (PyCFunction)m_apply, METH_VARARGS, NULL},
    {"inspect", (PyCFunction)m_inspect, METH_VARARGS, NULL},
    {"build", (PyCFunction)m_build, METH_NOARGS, NULL},
    {"roundtrip", (PyCFunction)m_roundtrip, METH_VARARGS, NULL},
    {"numbers", (PyCFunction)m_numbers, METH_VARARGS, NULL},
    {"dict_ops", (PyCFunction)m_dict_ops, METH_NOARGS, NULL},
    {"sequences", (PyCFunction)m_sequences, METH_NOARGS, NULL},
    {"singletons", (PyCFunction)m_singletons, METH_NOARGS, NULL},
    {"predicates", (PyCFunction)m_predicates, METH_VARARGS, NULL},
    {"fail", (PyCFunction)m_fail, METH_VARARGS, NULL},
    {"caught", (PyCFunction)m_caught, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static int methods_exec(PyObject *module)
{
    methods_state *state = state_of(module);
    if (state == NULL) {
        return -1;
    }
    state->calls = 0;
    if (PyModule_GetDef(module) != &moduledef) {
        PyErr_SetString(PyExc_SystemError, "PyModule_GetDef returned another definition");
        return -1;
    }
    if (PyModule_AddIntConstant(module, "ANSWER", 42) < 0) {
        return -1;
    }
    if (PyModule_AddStringConstant(module, "GREETING", "hi") < 0) {
        return -1;
    }
    PyObject *owned = PyLong_FromLong(11);
    if (owned == NULL) {
        return -1;
    }
    if (PyModule_AddObject(module, "OWNED", owned) < 0) {
        Py_DECREF(owned);
        return -1;
    }
    PyObject *shared = PyLong_FromLong(12);
    if (shared == NULL) {
        return -1;
    }
    int added = PyModule_AddObjectRef(module, "SHARED", shared);
    Py_DECREF(shared);
    if (added < 0) {
        return -1;
    }
    PyObject *namespace_ = PyModule_GetDict(module);
    if (namespace_ == NULL) {
        return -1;
    }
    PyObject *marker = PyUnicode_FromString("through-dict");
    if (marker == NULL) {
        return -1;
    }
    added = PyDict_SetItemString(namespace_, "VIA_DICT", marker);
    Py_DECREF(marker);
    return added < 0 ? -1 : 0;
}

static PyModuleDef_Slot slots[] = {
    {Py_mod_exec, (void *)methods_exec},
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpyext_methods",
    "pyre cpyext method module",
    sizeof(methods_state),
    methods,
    slots,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit_cpyext_methods(void)
{
    return PyModuleDef_Init(&moduledef);
}
