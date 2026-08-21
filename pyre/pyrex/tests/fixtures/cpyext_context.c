/* The interpreter state a call runs inside: the namespace a name falls back
   to, the context variables, and reporting a failure the caller has no way to
   hand back. */

#include <Python.h>

#include <string.h>

/* ── the fallback namespace ───────────────────────────────────────────── */

/* Borrowed, so a reference of this caller's own goes back to Python. */
static PyObject *builtins(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    PyObject *namespace = PyEval_GetBuiltins();
    if (namespace == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "no builtins namespace");
        return NULL;
    }
    return Py_NewRef(namespace);
}

/* ── the context variables ────────────────────────────────────────────── */

/* `ContextVar(name)`, or `ContextVar(name, default=...)` when one is given. */
static PyObject *var_new(PyObject *self, PyObject *args)
{
    (void)self;
    const char *name;
    PyObject *fallback = NULL;
    if (!PyArg_ParseTuple(args, "s|O", &name, &fallback)) {
        return NULL;
    }
    return PyContextVar_New(name, fallback);
}

static PyObject *var_set(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *variable;
    PyObject *value;
    if (!PyArg_ParseTuple(args, "OO", &variable, &value)) {
        return NULL;
    }
    return PyContextVar_Set(variable, value);
}

static PyObject *var_reset(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *variable;
    PyObject *token;
    if (!PyArg_ParseTuple(args, "OO", &variable, &token)) {
        return NULL;
    }
    if (PyContextVar_Reset(variable, token) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* `('unset',)` where the variable has neither a value here nor a default, and
   `('value', v)` where it has one -- the two a null `value` slot and a zero
   return distinguish. */
static PyObject *var_get(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *variable;
    PyObject *fallback = NULL;
    if (!PyArg_ParseTuple(args, "O|O", &variable, &fallback)) {
        return NULL;
    }
    PyObject *value = NULL;
    if (PyContextVar_Get(variable, fallback, &value) < 0) {
        return NULL;
    }
    if (value == NULL) {
        return Py_BuildValue("(s)", "unset");
    }
    PyObject *row = Py_BuildValue("(sO)", "value", value);
    Py_DECREF(value);
    return row;
}

/* ── reporting a failure ──────────────────────────────────────────────── */

/* Raise `ValueError(message)` and report it.  `which` picks between the
   spelling that takes the flag and the one that is `PyErr_PrintEx(1)`. */
static PyObject *report(PyObject *self, PyObject *args)
{
    (void)self;
    const char *which;
    const char *message;
    int set_last;
    if (!PyArg_ParseTuple(args, "ssi", &which, &message, &set_last)) {
        return NULL;
    }
    PyErr_SetString(PyExc_ValueError, message);
    if (strcmp(which, "print") == 0) {
        PyErr_Print();
    } else {
        PyErr_PrintEx(set_last);
    }
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Reporting with nothing to report, which is the caller's own mistake. */
static PyObject *report_nothing(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    PyErr_PrintEx(1);
    PyObject *raised = PyErr_GetRaisedException();
    if (raised == NULL) {
        Py_RETURN_NONE;
    }
    PyObject *name = PyUnicode_FromString(Py_TYPE(raised)->tp_name);
    Py_DECREF(raised);
    return name;
}

static PyMethodDef methods[] = {
    {"builtins", builtins, METH_NOARGS, NULL},
    {"var_new", var_new, METH_VARARGS, NULL},
    {"var_set", var_set, METH_VARARGS, NULL},
    {"var_reset", var_reset, METH_VARARGS, NULL},
    {"var_get", var_get, METH_VARARGS, NULL},
    {"report", report, METH_VARARGS, NULL},
    {"report_nothing", report_nothing, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_context", NULL, -1,
                                 methods};

PyMODINIT_FUNC PyInit_cpyext_context(void)
{
    return PyModule_Create(&def);
}
