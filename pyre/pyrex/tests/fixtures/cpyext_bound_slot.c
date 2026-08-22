/* Reading a slot off a type and calling it with an instance of something
   derived from that type -- what a compiled subclass does to reach the
   implementation it inherits.

   The types here are the ones this runtime defines, reached without a data
   symbol (none is exported) by taking `Py_TYPE` of an instance. */
#include <Python.h>

static PyTypeObject *dict_type;
static PyTypeObject *list_type;

/* `type->tp_repr(o)`, for a `type` the caller names rather than `Py_TYPE(o)`. */
static PyObject *repr_through(PyObject *self, PyObject *o)
{
    reprfunc slot = (reprfunc)PyType_GetSlot(dict_type, Py_tp_repr);
    (void)self;
    if (slot == NULL) {
        PyErr_SetString(PyExc_AssertionError, "dict carries no tp_repr");
        return NULL;
    }
    return slot(o);
}

static PyObject *length_through(PyObject *self, PyObject *o)
{
    lenfunc slot = (lenfunc)PyType_GetSlot(dict_type, Py_mp_length);
    Py_ssize_t counted;
    (void)self;
    if (slot == NULL) {
        PyErr_SetString(PyExc_AssertionError, "dict carries no mp_length");
        return NULL;
    }
    counted = slot(o);
    if (counted < 0) {
        return NULL;
    }
    return PyLong_FromSsize_t(counted);
}

static PyObject *item_through(PyObject *self, PyObject *args)
{
    ssizeargfunc slot = (ssizeargfunc)PyType_GetSlot(list_type, Py_sq_item);
    PyObject *o;
    Py_ssize_t at;
    (void)self;
    if (!PyArg_ParseTuple(args, "On", &o, &at)) {
        return NULL;
    }
    if (slot == NULL) {
        PyErr_SetString(PyExc_AssertionError, "list carries no sq_item");
        return NULL;
    }
    return slot(o, at);
}

/* Whether the two types answer a slot with the same function.  A slot bound
   to the type that owns the method is one function per owner, so two types
   that inherit the same method from the same owner share it and two that
   define their own do not. */
static PyObject *same_repr_slot(PyObject *self, PyObject *args)
{
    PyObject *first, *second;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &first, &second)) {
        return NULL;
    }
    if (!PyType_Check(first) || !PyType_Check(second)) {
        PyErr_SetString(PyExc_TypeError, "not a type");
        return NULL;
    }
    return PyBool_FromLong(
        PyType_GetSlot((PyTypeObject *)first, Py_tp_repr)
        == PyType_GetSlot((PyTypeObject *)second, Py_tp_repr));
}

static PyMethodDef methods[] = {
    {"repr_through", repr_through, METH_O, NULL},
    {"length_through", length_through, METH_O, NULL},
    {"item_through", item_through, METH_VARARGS, NULL},
    {"same_repr_slot", same_repr_slot, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "cpyext_bound_slot",
    "pyre cpyext bound-slot module", -1, methods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_cpyext_bound_slot(void)
{
    PyObject *mapping = PyDict_New();
    PyObject *sequence = PyList_New(0);
    if (mapping == NULL || sequence == NULL) {
        Py_XDECREF(mapping);
        Py_XDECREF(sequence);
        return NULL;
    }
    dict_type = Py_TYPE(mapping);
    Py_INCREF(dict_type);
    Py_DECREF(mapping);
    list_type = Py_TYPE(sequence);
    Py_INCREF(list_type);
    Py_DECREF(sequence);
    return PyModule_Create(&moduledef);
}
