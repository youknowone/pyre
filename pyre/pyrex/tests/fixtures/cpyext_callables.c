/* The callables an extension makes and asks about: the `PyMethodDef`-backed
   function, the bound method, and the reports a call that cannot raise leaves
   behind. */
#include "Python.h"

static PyObject *probe(PyObject *self, PyObject *arg)
{
    (void)self;
    Py_INCREF(arg);
    return arg;
}

static PyMethodDef probe_def = {"probe", probe, METH_O, NULL};

/* Whether each argument is backed by a C function. */
static PyObject *is_c_function(PyObject *self, PyObject *arg)
{
    (void)self;
    return PyLong_FromLong(PyCFunction_Check(arg));
}

/* Whether `arg` is this module's own `probe`, compared the way an extension
   recognises one of its own: the C function behind it. */
static PyObject *is_probe(PyObject *self, PyObject *arg)
{
    (void)self;
    if (!PyCFunction_Check(arg)) {
        return PyLong_FromLong(0);
    }
    return PyLong_FromLong(PyCFunction_GET_FUNCTION(arg) == (PyCFunction)probe);
}

/* The flags and receiver the carrier holds, beside the name it answers to. */
static PyObject *describe(PyObject *self, PyObject *arg)
{
    PyObject *receiver;
    (void)self;
    if (!PyCFunction_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "not a C function");
        return NULL;
    }
    receiver = PyCFunction_GET_SELF(arg);
    if (receiver == NULL) {
        receiver = Py_None;
    }
    return Py_BuildValue("iO", PyCFunction_GET_FLAGS(arg), receiver);
}

/* A fresh callable over the same definition, bound to `receiver`. */
static PyObject *make_c_function(PyObject *self, PyObject *receiver)
{
    (void)self;
    return PyCFunction_New(&probe_def, receiver);
}

/* `func.__get__(receiver)` spelled the way C spells it, then read back. */
static PyObject *bind(PyObject *self, PyObject *args)
{
    PyObject *function;
    PyObject *receiver;
    PyObject *method;
    PyObject *answer;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &function, &receiver)) {
        return NULL;
    }
    method = PyMethod_New(function, receiver);
    if (method == NULL) {
        return NULL;
    }
    answer = Py_BuildValue("OiOO", method, PyMethod_Check(method),
                           PyMethod_GET_FUNCTION(method), PyMethod_GET_SELF(method));
    Py_DECREF(method);
    return answer;
}

/* `sys.modules`, which an extension reaches without importing `sys`. */
static PyObject *module_dict(PyObject *self, PyObject *unused)
{
    PyObject *modules = PyImport_GetModuleDict();
    (void)self;
    (void)unused;
    Py_XINCREF(modules);
    return modules;
}

/* `__import__` with every argument stated. */
static PyObject *import_level(PyObject *self, PyObject *args)
{
    const char *name;
    PyObject *fromlist;
    int level;
    (void)self;
    if (!PyArg_ParseTuple(args, "sOi", &name, &fromlist, &level)) {
        return NULL;
    }
    return PyImport_ImportModuleLevel(name, NULL, NULL, fromlist, level);
}

/* The borrowed spelling of `setdefault`. */
static PyObject *set_default(PyObject *self, PyObject *args)
{
    PyObject *dict;
    PyObject *key;
    PyObject *value;
    PyObject *answer;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOO", &dict, &key, &value)) {
        return NULL;
    }
    answer = PyDict_SetDefault(dict, key, value);
    if (answer == NULL) {
        return NULL;
    }
    Py_INCREF(answer);
    return answer;
}

/* A call whose keywords arrive as a mapping rather than beside the values. */
static PyObject *call_with_dict(PyObject *self, PyObject *args)
{
    PyObject *callable;
    PyObject *positional;
    PyObject *keywords;
    PyObject *vector[3];
    Py_ssize_t count;
    Py_ssize_t index;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO!O!", &callable, &PyTuple_Type, &positional,
                          &PyDict_Type, &keywords)) {
        return NULL;
    }
    count = PyTuple_Size(positional);
    if (count > 3) {
        PyErr_SetString(PyExc_ValueError, "at most three positional arguments");
        return NULL;
    }
    for (index = 0; index < count; index++) {
        vector[index] = PyTuple_GetItem(positional, index);
    }
    return PyObject_VectorcallDict(callable, vector, (size_t)count, keywords);
}

/* Report the pending exception the way a caller that cannot raise does. */
static PyObject *report_unraisable(PyObject *self, PyObject *arg)
{
    (void)self;
    PyErr_SetString(PyExc_ValueError, "boom");
    PyErr_WriteUnraisable(arg);
    Py_RETURN_NONE;
}

/* The same, stating what was going on instead of naming an object. */
static PyObject *report_unraisable_msg(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    PyErr_SetString(PyExc_ValueError, "boom");
    PyErr_FormatUnraisable("Exception ignored while doing %s", "the thing");
    Py_RETURN_NONE;
}

/* The version the extension is running against, and the interpreter it is in. */
static PyObject *runtime_identity(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    return Py_BuildValue("kL", Py_Version,
                         PyInterpreterState_GetID(PyThreadState_Get()->interp));
}

/* Whether the argument is a traceback. */
static PyObject *is_traceback(PyObject *self, PyObject *arg)
{
    (void)self;
    return PyLong_FromLong(PyTraceBack_Check(arg));
}

static PyMethodDef methods[] = {
    {"is_c_function", is_c_function, METH_O, NULL},
    {"is_probe", is_probe, METH_O, NULL},
    {"describe", describe, METH_O, NULL},
    {"make_c_function", make_c_function, METH_O, NULL},
    {"probe", probe, METH_O, NULL},
    {"bind", bind, METH_VARARGS, NULL},
    {"module_dict", module_dict, METH_NOARGS, NULL},
    {"import_level", import_level, METH_VARARGS, NULL},
    {"set_default", set_default, METH_VARARGS, NULL},
    {"call_with_dict", call_with_dict, METH_VARARGS, NULL},
    {"report_unraisable", report_unraisable, METH_O, NULL},
    {"report_unraisable_msg", report_unraisable_msg, METH_NOARGS, NULL},
    {"runtime_identity", runtime_identity, METH_NOARGS, NULL},
    {"is_traceback", is_traceback, METH_O, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_callables", NULL,
                                 -1, methods};

PyMODINIT_FUNC PyInit_cpyext_callables(void)
{
    return PyModule_Create(&def);
}
