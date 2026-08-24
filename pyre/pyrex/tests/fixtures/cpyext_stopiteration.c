/* The one exception field an extension reads out of the block:
   `__Pyx_PyGen_FetchStopIterationValue` takes what a generator returned
   straight out of `PyStopIterationObject.value`, which is what every Cython
   module with a `yield` in it compiles a `for` over a generator into. */
#include "Python.h"

/* The layout, stated here so a change on the Rust side that does not reach the
   header stops this fixture compiling. */
static PyObject *layout(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    return Py_BuildValue("nn", (Py_ssize_t)offsetof(PyStopIterationObject, value),
                         (Py_ssize_t)sizeof(PyStopIterationObject));
}

/* The value read through the block, beside the one the attribute answers. */
static PyObject *value_of(PyObject *self, PyObject *exception)
{
    PyObject *through_block;
    PyObject *through_attribute;
    PyObject *pair;
    (void)self;
    if (!PyObject_TypeCheck(exception, (PyTypeObject *)PyExc_StopIteration)) {
        PyErr_SetString(PyExc_TypeError, "not a StopIteration");
        return NULL;
    }
    through_block = ((PyStopIterationObject *)exception)->value;
    if (through_block == NULL) {
        PyErr_SetString(PyExc_AssertionError, "value is NULL");
        return NULL;
    }
    through_attribute = PyObject_GetAttrString(exception, "value");
    if (through_attribute == NULL) {
        return NULL;
    }
    pair = Py_BuildValue("OO", through_block, through_attribute);
    Py_DECREF(through_attribute);
    return pair;
}

/* The sequence Cython runs: drive an iterator to exhaustion through
   `__next__`, then take the returned value out of the exception the fetch
   handed over.  `PyIter_Next` swallows the `StopIteration` and the value with
   it, so the value has to be reached through a call that lets it out.
   `PyErr_NormalizeException` is what turns the raised class into the instance
   the read needs. */
static PyObject *fetch_returned(PyObject *self, PyObject *iterator)
{
    PyObject *item;
    PyObject *type, *value, *traceback;
    PyObject *returned;
    (void)self;
    while ((item = PyObject_CallMethod(iterator, "__next__", NULL)) != NULL) {
        Py_DECREF(item);
    }
    if (!PyErr_ExceptionMatches(PyExc_StopIteration)) {
        return NULL;
    }
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    /* The normalize is what makes the block readable, and the read is a field
       that a `StopIteration` raised bare leaves unset. */
    returned = NULL;
    if (value != NULL
        && PyObject_TypeCheck(value, (PyTypeObject *)PyExc_StopIteration)) {
        returned = ((PyStopIterationObject *)value)->value;
    }
    if (returned == NULL) {
        PyErr_SetString(PyExc_AssertionError, "the fetch carried no value");
    } else {
        Py_INCREF(returned);
    }
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    return returned;
}

/* A class derived from `StopIteration` in C: its storage begins with the
   base's, so the same read reaches the same word and the field it declares
   past the end is addressable without disturbing it.

   The type declares no `tp_init` of its own: an extension cannot reach a
   builtin exception's through `((PyTypeObject *)PyExc_StopIteration)->tp_init`
   here, because a class this runtime defines has a mirror whose slots are all
   null.  Leaving it unset inherits the base's constructor, which is what fills
   `value`. */
typedef struct {
    PyStopIterationObject base;
    long marker;
} DerivedObject;

static PyObject *derived_marker(PyObject *self, PyObject *exception)
{
    (void)self;
    return PyLong_FromLong(((DerivedObject *)exception)->marker);
}

static PyObject *set_derived_marker(PyObject *self, PyObject *args)
{
    PyObject *exception;
    long marker;
    (void)self;
    if (!PyArg_ParseTuple(args, "Ol", &exception, &marker)) {
        return NULL;
    }
    ((DerivedObject *)exception)->marker = marker;
    Py_RETURN_NONE;
}

static PyType_Slot derived_slots[] = {
    {0, NULL},
};

static PyType_Spec derived_spec = {
    "cpyext_stopiteration.Derived",
    sizeof(DerivedObject),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    derived_slots,
};

static PyMethodDef methods[] = {
    {"layout", layout, METH_NOARGS, NULL},
    {"value_of", value_of, METH_O, NULL},
    {"fetch_returned", fetch_returned, METH_O, NULL},
    {"derived_marker", derived_marker, METH_O, NULL},
    {"set_derived_marker", set_derived_marker, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static int stopiteration_exec(PyObject *module)
{
    PyObject *bases;
    PyObject *made;
    bases = PyTuple_Pack(1, PyExc_StopIteration);
    if (bases == NULL) {
        return -1;
    }
    made = PyType_FromModuleAndSpec(module, &derived_spec, bases);
    Py_DECREF(bases);
    if (made == NULL) {
        return -1;
    }
    if (PyModule_AddObjectRef(module, "Derived", made) < 0) {
        Py_DECREF(made);
        return -1;
    }
    Py_DECREF(made);
    return 0;
}

static PyModuleDef_Slot slots[] = {
    {Py_mod_exec, (void *)stopiteration_exec},
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpyext_stopiteration",
    "pyre cpyext StopIteration module",
    0,
    methods,
    slots,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit_cpyext_stopiteration(void)
{
    return PyModuleDef_Init(&moduledef);
}
